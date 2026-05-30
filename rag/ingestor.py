from __future__ import annotations

import logging
import re
from datetime import date, datetime
from pathlib import Path
from typing import ClassVar
from uuid import uuid4

import fitz
import pdfplumber

from rag.embedder import Embedder
from rag.retriever import HybridRetriever
from vectordb.chroma_client import ChromaClient

logger = logging.getLogger(__name__)


class DocumentIngestor:
	_valid_domains: ClassVar[tuple[str, ...]] = ("hr", "legal", "finance", "engineering")
	_domain_min_lengths: ClassVar[dict[str, int]] = {
		"hr": 100,
		"legal": 80,
		"finance": 60,
		"engineering": 80,
	}
	_domain_split_patterns: ClassVar[dict[str, re.Pattern[str] | None]] = {
		"hr": None,
		"legal": re.compile(r"\n(?=\d+\.\d*\s|\bArticle\b|\bClause\b)", re.IGNORECASE),
		"finance": re.compile(
			r"\n(?=(?:[A-Z][A-Z0-9 ,/&().\-]{4,}$)|(?:\d+\s))",
			re.MULTILINE,
		),
		"engineering": re.compile(r"\n(?=##\s|###\s|\d+\.\s[A-Z])"),
	}
	_max_chunk_length: ClassVar[int] = 1000

	def __init__(self) -> None:
		self._chroma_client = ChromaClient()
		self._embedder = Embedder()
		self._retriever = HybridRetriever()

	def parse_pdf(self, file_path: str) -> list[dict[str, object]]:
		path = Path(file_path)
		if not path.is_file():
			raise FileNotFoundError(f"File not found: {file_path}")

		pages: list[dict[str, object]] = []
		total_chars = 0
		plumber_pdf: pdfplumber.PDF | None = None

		try:
			with fitz.open(str(path)) as pdf_document:
				page_count = len(pdf_document)
				for index, page in enumerate(pdf_document, start=1):
					raw_text = self._strip_header_footer(page.get_text("text") or "")
					if not raw_text.strip():
						if plumber_pdf is None:
							plumber_pdf = pdfplumber.open(str(path))
						if index - 1 < len(plumber_pdf.pages):
							fallback_text = plumber_pdf.pages[index - 1].extract_text() or ""
							raw_text = self._strip_header_footer(fallback_text)

					raw_text = raw_text.strip()
					pages.append({"page_number": index, "raw_text": raw_text})
					total_chars += len(raw_text)

				logger.info(
					"Parsed PDF | file=%s | pages=%s | total_chars=%s",
					path.name,
					page_count,
					total_chars,
				)
		finally:
			if plumber_pdf is not None:
				plumber_pdf.close()

		if total_chars == 0:
			raise ValueError("No extractable text. Document may be scanned.")

		return pages

	def chunk_document(self, pages: list[dict[str, object]], domain: str) -> list[dict[str, object]]:
		normalized_domain = self._normalize_domain(domain)
		minimum_length = self._domain_min_lengths[normalized_domain]
		chunks: list[dict[str, object]] = []

		for page in pages:
			page_number = int(page.get("page_number", 0) or 0)
			page_text = str(page.get("raw_text", "") or "").strip()
			if not page_text:
				continue

			segments = self._split_page_text(page_text, normalized_domain)
			for segment in segments:
				text = segment.strip()
				if not text:
					continue

				if chunks and len(text) < minimum_length:
					chunks[-1]["text"] = f"{chunks[-1]['text']}\n\n{text}".strip()
					chunks[-1]["char_count"] = len(str(chunks[-1]["text"]))
					continue

				chunks.append(
					{
						"chunk_id": str(uuid4()),
						"text": text,
						"domain": normalized_domain,
						"page_number": page_number,
						"chunk_index": 0,
						"char_count": len(text),
					}
				)

		final_chunks: list[dict[str, object]] = []
		for chunk in chunks:
			chunk_text = str(chunk.get("text", "") or "").strip()
			if not chunk_text:
				continue

			for split_text in self._split_long_text(chunk_text):
				split_text = split_text.strip()
				if not split_text:
					continue

				final_chunks.append(
					{
						"chunk_id": str(uuid4()),
						"text": split_text,
						"domain": normalized_domain,
						"page_number": chunk.get("page_number", 0),
						"chunk_index": len(final_chunks),
						"char_count": len(split_text),
					}
				)

		avg_chunk_size = (
			sum(int(chunk["char_count"]) for chunk in final_chunks) / len(final_chunks)
			if final_chunks
			else 0.0
		)
		logger.info(
			"Chunked document | domain=%s | chunk_count=%s | avg_chunk_size=%.2f",
			normalized_domain,
			len(final_chunks),
			avg_chunk_size,
		)
		return final_chunks

	async def ingest_pdf(self, file_path: str, domain: str, metadata: dict[str, object]) -> dict[str, object]:
		normalized_domain = self._normalize_domain(domain)
		path = Path(file_path)
		if not path.is_file():
			raise FileNotFoundError(f"File not found: {file_path}")

		version = metadata.get("version")
		effective_date = metadata.get("effective_date")
		if version is None:
			raise ValueError("version is required")
		if effective_date is None:
			raise ValueError("effective_date is required")

		pages = self.parse_pdf(str(path))
		chunks = self.chunk_document(pages, normalized_domain)
		if not chunks:
			raise ValueError("No extractable text. Document may be scanned.")

		texts = [str(chunk["text"]) for chunk in chunks]
		embeddings = self._embedder.embed_batch(texts)
		if len(embeddings) != len(chunks):
			raise RuntimeError("Embedding batch size mismatch")

		source_file = path.name
		ingested_at = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
		prepared_metadatas = [
			self._prepare_metadata(
				chunk=chunk,
				source_file=source_file,
				domain=normalized_domain,
				version=version,
				effective_date=effective_date,
				ingested_at=ingested_at,
			)
			for chunk in chunks
		]
		ids = [str(chunk["chunk_id"]) for chunk in chunks]

		collection = self._chroma_client.get_collection(normalized_domain)
		self._chroma_client.delete_chunks_by_source(normalized_domain, source_file)
		pre_add_count = collection.count()

		try:
			collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=prepared_metadatas)
		except Exception as exc:
			post_add_count = collection.count()
			succeeded = max(post_add_count - pre_add_count, 0)
			logger.exception(
				"ChromaDB insertion failed | domain=%s | file=%s | succeeded=%s | total=%s",
				normalized_domain,
				source_file,
				succeeded,
				len(chunks),
			)
			raise RuntimeError(
				f"ChromaDB insertion failure after {succeeded} of {len(chunks)} chunks succeeded"
			) from exc

		self._retriever.invalidate_bm25_index(normalized_domain)
		logger.info(
			"Ingested PDF | file=%s | domain=%s | chunks=%s",
			source_file,
			normalized_domain,
			len(chunks),
		)
		return {
			"chunks_ingested": len(chunks),
			"domain": normalized_domain,
			"source_file": source_file,
			"version": version,
			"effective_date": self._serialize_metadata_value(effective_date),
			"ingested_at": ingested_at,
		}

	def _split_page_text(self, text: str, domain: str) -> list[str]:
		if domain == "hr":
			segments = text.split("\n\n")
		else:
			pattern = self._domain_split_patterns[domain]
			segments = pattern.split(text) if pattern is not None else [text]

		return [segment.strip() for segment in segments if segment and segment.strip()]

	def _split_long_text(self, text: str) -> list[str]:
		if len(text) <= self._max_chunk_length:
			return [text]

		sentences = re.split(r"(?<=[.!?])\s+", text)
		if len(sentences) == 1:
			return self._hard_split_text(text)

		chunks: list[str] = []
		current = ""
		for sentence in sentences:
			sentence = sentence.strip()
			if not sentence:
				continue

			candidate = sentence if not current else f"{current} {sentence}"
			if len(candidate) <= self._max_chunk_length:
				current = candidate
				continue

			if current:
				chunks.append(current)
				current = sentence
			else:
				chunks.extend(self._hard_split_text(sentence))
				current = ""

		if current:
			chunks.append(current)

		if any(len(chunk) > self._max_chunk_length for chunk in chunks):
			flattened: list[str] = []
			for chunk in chunks:
				if len(chunk) > self._max_chunk_length:
					flattened.extend(self._hard_split_text(chunk))
				else:
					flattened.append(chunk)
			return flattened

		return chunks

	def _hard_split_text(self, text: str) -> list[str]:
		words = text.split()
		if not words:
			return []

		chunks: list[str] = []
		current_words: list[str] = []
		for word in words:
			candidate_words = current_words + [word]
			candidate = " ".join(candidate_words)
			if len(candidate) <= self._max_chunk_length:
				current_words = candidate_words
				continue

			if current_words:
				chunks.append(" ".join(current_words))
				current_words = [word]
			else:
				chunks.extend(self._split_overlong_token(word))
				current_words = []

		if current_words:
			chunks.append(" ".join(current_words))

		return chunks

	def _split_overlong_token(self, token: str) -> list[str]:
		if len(token) <= self._max_chunk_length:
			return [token]

		return [token[index : index + self._max_chunk_length] for index in range(0, len(token), self._max_chunk_length)]

	@staticmethod
	def _strip_header_footer(text: str) -> str:
		lines = text.splitlines()
		if not lines:
			return ""

		if len(lines) > 0 and len(lines[0].strip()) < 60:
			lines = lines[1:]
		if lines and len(lines[-1].strip()) < 60:
			lines = lines[:-1]
		return "\n".join(lines)

	def _prepare_metadata(
		self,
		chunk: dict[str, object],
		source_file: str,
		domain: str,
		version: object,
		effective_date: object,
		ingested_at: str,
	) -> dict[str, object]:
		return {
			"source_file": source_file,
			"domain": domain,
			"page_number": int(chunk.get("page_number", 0) or 0),
			"chunk_index": int(chunk.get("chunk_index", 0) or 0),
			"char_count": int(chunk.get("char_count", 0) or 0),
			"version": self._serialize_metadata_value(version),
			"effective_date": self._serialize_metadata_value(effective_date),
			"ingested_at": ingested_at,
		}

	@classmethod
	def _normalize_domain(cls, domain: str) -> str:
		normalized = str(domain or "").strip().lower()
		if normalized not in cls._valid_domains:
			raise ValueError("Unsupported domain")
		return normalized

	@staticmethod
	def _serialize_metadata_value(value: object) -> str:
		if isinstance(value, datetime):
			return value.isoformat()
		if isinstance(value, date):
			return value.isoformat()
		return str(value)


__all__ = ["DocumentIngestor"]

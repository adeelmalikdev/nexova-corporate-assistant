from __future__ import annotations

import logging
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import ClassVar

import chromadb

from config import settings

logger = logging.getLogger(__name__)


class ChromaClient:
	_instance: ClassVar[ChromaClient | None] = None
	_initialized: ClassVar[bool] = False
	_collection_cache: ClassVar[dict[str, chromadb.Collection]] = {}
	_collection_lock: ClassVar[threading.Lock] = threading.Lock()
	_required_metadata_fields: ClassVar[tuple[str, ...]] = (
		"source_file",
		"domain",
		"page_number",
		"chunk_index",
		"version",
		"effective_date",
		"ingested_at",
	)

	def __new__(cls) -> ChromaClient:
		if cls._instance is None:
			cls._instance = super().__new__(cls)
		return cls._instance

	def __init__(self) -> None:
		if self.__class__._initialized:
			return

		persist_path = Path(settings.CHROMA_PERSIST_PATH)
		persist_path.mkdir(parents=True, exist_ok=True)
		self._client = chromadb.PersistentClient(path=str(persist_path))
		self.__class__._initialized = True
		logger.info("Initialized ChromaClient at %s", persist_path)

	def ping(self) -> bool:
		try:
			self._client.heartbeat()
			return True
		except Exception:
			logger.exception("ChromaDB heartbeat failed")
			return False

	@staticmethod
	def _normalize_domain(domain: str) -> str:
		normalized = re.sub(r"[^a-z0-9]+", "_", domain.strip().lower()).strip("_")
		if not normalized:
			raise ValueError("domain must not be empty")
		return normalized

	def _collection_name(self, domain: str) -> str:
		return f"nexova_{self._normalize_domain(domain)}"

	def _collection_metadata(self) -> dict[str, str]:
		return {
			"hnsw:space": "cosine",
			"metadata_schema": ",".join(self._required_metadata_fields),
		}

	def get_collection(self, domain: str) -> chromadb.Collection:
		logger.info("Getting collection for domain: %s", domain)
		collection_name = self._collection_name(domain)

		with self._collection_lock:
			collection = self._collection_cache.get(collection_name)
			if collection is None:
				collection = self._client.get_or_create_collection(
					name=collection_name,
					metadata=self._collection_metadata(),
				)
				self._collection_cache[collection_name] = collection
				logger.info("Created or loaded collection: %s", collection_name)
			else:
				logger.info("Reusing cached collection: %s", collection_name)

		current_metadata = collection.metadata or {}
		desired_metadata = self._collection_metadata()
		if any(current_metadata.get(key) != value for key, value in desired_metadata.items()):
			try:
				collection.modify(metadata=desired_metadata)
				logger.info("Updated collection metadata for %s", collection_name)
			except Exception:
				logger.exception("Failed to enforce collection metadata for %s", collection_name)

		return collection

	def delete_chunks_by_source(self, domain: str, source_file: str) -> int:
		logger.info("Deleting chunks for domain=%s source_file=%s", domain, source_file)
		collection = self.get_collection(domain)
		result = collection.get(where={"source_file": source_file}, include=[])
		chunk_ids = result.get("ids", [])

		if not chunk_ids:
			logger.info("No chunks found for domain=%s source_file=%s", domain, source_file)
			return 0

		collection.delete(ids=chunk_ids)
		deleted_count = len(chunk_ids)
		logger.info(
			"Deleted %s chunks for domain=%s source_file=%s",
			deleted_count,
			domain,
			source_file,
		)
		return deleted_count

	def get_collection_stats(self, domain: str) -> dict[str, str | int | None]:
		logger.info("Fetching collection stats for domain: %s", domain)
		collection = self.get_collection(domain)
		chunk_count = collection.count()

		last_ingested: datetime | None = None
		offset = 0
		batch_size = 1000
		while True:
			result = collection.get(limit=batch_size, offset=offset, include=["metadatas"])
			metadatas = result.get("metadatas", [])
			if not metadatas:
				break

			for metadata in metadatas:
				if not metadata:
					continue
				ingested_at = metadata.get("ingested_at")
				if not isinstance(ingested_at, str) or not ingested_at:
					continue
				try:
					parsed = datetime.fromisoformat(ingested_at.replace("Z", "+00:00"))
				except ValueError:
					logger.info("Skipping unparseable ingested_at value: %s", ingested_at)
					continue
				if last_ingested is None or parsed > last_ingested:
					last_ingested = parsed

			if len(result.get("ids", [])) < batch_size:
				break
			offset += batch_size

		stats = {
			"domain": self._normalize_domain(domain),
			"chunk_count": chunk_count,
			"last_ingested": last_ingested.isoformat() if last_ingested else None,
		}
		logger.info("Collection stats for %s: %s", domain, stats)
		return stats

	def get_all_chunks(self, domain: str) -> list[dict[str, object]]:
		logger.info("Fetching all chunks for domain: %s", domain)
		collection = self.get_collection(domain)
		batch_size = 1000
		offset = 0
		chunks: list[dict[str, object]] = []

		while True:
			result = collection.get(limit=batch_size, offset=offset, include=["documents", "metadatas"])
			ids = result.get("ids", [])
			documents = result.get("documents", [])
			metadatas = result.get("metadatas", [])

			if not ids:
				break

			for chunk_id, document, metadata in zip(ids, documents, metadatas):
				chunks.append(
					{
						"id": chunk_id,
						"text": document,
						"metadata": metadata or {},
					}
				)

			if len(ids) < batch_size:
				break
			offset += batch_size

		chunks.sort(
			key=lambda item: (
				item["metadata"].get("page_number", 0) if isinstance(item["metadata"], dict) else 0,
				item["metadata"].get("chunk_index", 0) if isinstance(item["metadata"], dict) else 0,
				str(item["id"]),
			)
		)
		logger.info("Fetched %s chunks for domain=%s", len(chunks), domain)
		return chunks

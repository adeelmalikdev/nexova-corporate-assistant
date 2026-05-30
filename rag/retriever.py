from __future__ import annotations

import asyncio
import logging
import re
import threading
from typing import ClassVar

from rank_bm25 import BM25Okapi

from config import settings
from rag.embedder import Embedder
from rag.reranker import Reranker
from vectordb.chroma_client import ChromaClient

logger = logging.getLogger(__name__)


class HybridRetriever:
	_bm25_indexes: ClassVar[dict[str, BM25Okapi]] = {}
	_bm25_chunks: ClassVar[dict[str, list[dict[str, object]]]] = {}
	_bm25_lock: ClassVar[threading.Lock] = threading.Lock()

	def __init__(self) -> None:
		self._chroma_client = ChromaClient()
		self._embedder = Embedder()
		self._reranker = Reranker()

	def build_bm25_index(self, domain: str) -> None:
		with self._bm25_lock:
			chunks = self._chroma_client.get_all_chunks(domain)
			self.__class__._bm25_chunks[domain] = chunks
			if not chunks:
				self.__class__._bm25_indexes.pop(domain, None)
				logger.info("Built empty BM25 index for domain=%s", domain)
				return

			tokenized_corpus = [self._tokenize(chunk.get("text", "")) for chunk in chunks]
			self.__class__._bm25_indexes[domain] = BM25Okapi(tokenized_corpus)
			logger.info("Built BM25 index for domain=%s with %s chunks", domain, len(chunks))

	def invalidate_bm25_index(self, domain: str) -> None:
		with self._bm25_lock:
			self.__class__._bm25_indexes.pop(domain, None)
			self.__class__._bm25_chunks.pop(domain, None)
			logger.info("Invalidated BM25 index for domain=%s", domain)

	async def dense_search(
		self,
		query: str,
		domain: str,
		top_k: int,
		filters: dict | None = None,
	) -> list[dict[str, object]]:
		if top_k <= 0:
			return []

		query_embedding = self._embedder.embed_query(query)
		collection = self._chroma_client.get_collection(domain)
		where = self._build_where_clause(filters)

		query_kwargs: dict[str, object] = {
			"query_embeddings": [query_embedding],
			"n_results": top_k,
			"include": ["documents", "metadatas", "distances"],
		}
		if where:
			query_kwargs["where"] = where

		result = collection.query(**query_kwargs)
		ids = result.get("ids", [[]])
		documents = result.get("documents", [[]])
		metadatas = result.get("metadatas", [[]])
		distances = result.get("distances", [[]])

		chunks: list[dict[str, object]] = []
		for chunk_id, document, metadata, distance in zip(ids[0], documents[0], metadatas[0], distances[0]):
			distance_value = float(distance or 0.0)
			similarity_score = 1.0 - distance_value
			if similarity_score < settings.SIMILARITY_THRESHOLD:
				continue

			chunk = {
				"id": chunk_id,
				"chunk_id": chunk_id,
				"text": document or "",
				"metadata": metadata or {},
				"distance": distance_value,
				"similarity_score": similarity_score,
			}
			chunks.append(chunk)

		logger.info("Dense search returned %s chunks for domain=%s", len(chunks), domain)
		return chunks

	def bm25_search(self, query: str, domain: str, top_k: int) -> list[dict[str, object]]:
		if top_k <= 0:
			return []

		if domain not in self._bm25_indexes or domain not in self._bm25_chunks:
			self.build_bm25_index(domain)

		index = self._bm25_indexes.get(domain)
		chunks = self._bm25_chunks.get(domain, [])
		if index is None or not chunks:
			return []

		query_tokens = self._tokenize(query)
		if not query_tokens:
			return []

		scores = index.get_scores(query_tokens)
		score_pairs = list(zip(chunks, scores))
		if not score_pairs:
			return []

		max_score = max(float(score) for _, score in score_pairs)
		if max_score <= 0.0:
			normalized_scores = [(chunk, 0.0) for chunk, _ in score_pairs]
		else:
			normalized_scores = [(chunk, float(score) / max_score) for chunk, score in score_pairs]

		ranked = sorted(normalized_scores, key=lambda item: item[1], reverse=True)[:top_k]
		results: list[dict[str, object]] = []
		for chunk, score in ranked:
			result = dict(chunk)
			result["bm25_score"] = float(score)
			results.append(result)

		logger.info("BM25 search returned %s chunks for domain=%s", len(results), domain)
		return results

	def reciprocal_rank_fusion(
		self,
		dense_results: list[dict[str, object]],
		bm25_results: list[dict[str, object]],
		k: int = 60,
	) -> list[dict[str, object]]:
		merged: dict[str, dict[str, object]] = {}

		for rank, chunk in enumerate(dense_results, start=1):
			self._merge_rrf_chunk(
				merged,
				chunk,
				weight=settings.DENSE_WEIGHT,
				rank=rank,
				k=k,
			)

		for rank, chunk in enumerate(bm25_results, start=1):
			self._merge_rrf_chunk(
				merged,
				chunk,
				weight=settings.BM25_WEIGHT,
				rank=rank,
				k=k,
			)

		results = list(merged.values())
		results.sort(key=lambda item: float(item.get("rrf_score", 0.0)), reverse=True)
		return results

	async def retrieve(
		self,
		query: str,
		domain: str,
		top_k_final: int = 3,
		filters: dict | None = None,
	) -> list[dict[str, object]]:
		candidates = await self._retrieve_candidates(query, domain, filters)
		if not candidates:
			return []
		return self._reranker.rerank(query, candidates, top_k_final)

	async def retrieve_multi_domain(
		self,
		queries: list[str],
		domains: list[str],
		top_k_final: int = 3,
		filters: dict | None = None,
	) -> dict[str, list[dict[str, object]]]:
		if not domains:
			return {}
		if not queries:
			return {domain: [] for domain in domains}

		async def retrieve_domain(domain: str) -> list[dict[str, object]]:
			query_tasks = [self._retrieve_candidates(query, domain, filters) for query in queries]
			query_results = await asyncio.gather(*query_tasks, return_exceptions=True)
			chunk_lists = [result for result in query_results if not isinstance(result, Exception)]
			merged = self._merge_chunk_lists(chunk_lists)
			if not merged:
				return []
			return self._reranker.rerank(queries[0], merged, top_k_final)

		domain_tasks = [retrieve_domain(domain) for domain in domains]
		domain_results = await asyncio.gather(*domain_tasks, return_exceptions=True)

		results: dict[str, list[dict[str, object]]] = {}
		for domain, result in zip(domains, domain_results):
			if isinstance(result, Exception):
				logger.exception("Failed multi-domain retrieval for domain=%s", domain)
				results[domain] = []
			else:
				results[domain] = result
		return results

	async def _retrieve_candidates(
		self,
		query: str,
		domain: str,
		filters: dict | None = None,
	) -> list[dict[str, object]]:
		dense_task = self.dense_search(query, domain, settings.TOP_K_RETRIEVAL, filters)
		bm25_task = asyncio.to_thread(self.bm25_search, query, domain, settings.TOP_K_RETRIEVAL)
		dense_results, bm25_results = await asyncio.gather(dense_task, bm25_task)
		merged = self.reciprocal_rank_fusion(dense_results, bm25_results)
		return merged[: settings.TOP_K_RETRIEVAL]

	@staticmethod
	def format_context(chunks: list[dict[str, object]]) -> str:
		parts: list[str] = []
		for chunk in chunks:
			metadata = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
			source_file = str(metadata.get("source_file", "unknown"))
			page_number = metadata.get("page_number", "unknown")
			text = str(chunk.get("text", "") or "").strip()
			rerank_score = float(chunk.get("rerank_score", 0.0))
			parts.append(
				f"[SOURCE: {source_file} | Page {page_number} | Relevance: {rerank_score:.2f}]\n"
				f"{text}\n"
				"---"
			)
		return "\n".join(parts)

	@staticmethod
	def _tokenize(text: object) -> list[str]:
		return re.findall(r"\b\w+\b", str(text or "").lower())

	@staticmethod
	def _build_where_clause(filters: dict | None) -> dict[str, object] | None:
		if not filters:
			return None

		where: dict[str, object] = {}
		for key, value in filters.items():
			if value is None:
				continue
			if key == "effective_date":
				where[key] = {"$gte": HybridRetriever._serialize_filter_value(value)}
			else:
				where[key] = value

		return where or None

	@staticmethod
	def _serialize_filter_value(value: object) -> str:
		if hasattr(value, "isoformat"):
			return str(value.isoformat())
		return str(value)

	@staticmethod
	def _chunk_id(chunk: dict[str, object]) -> str:
		chunk_id = chunk.get("chunk_id") or chunk.get("id")
		if chunk_id is None:
			return ""
		return str(chunk_id)

	@classmethod
	def _merge_rrf_chunk(
		cls,
		merged: dict[str, dict[str, object]],
		chunk: dict[str, object],
		weight: float,
		rank: int,
		k: int,
	) -> None:
		chunk_id = cls._chunk_id(chunk)
		if not chunk_id:
			return

		contribution = weight * (1.0 / float(rank + k))
		existing = merged.get(chunk_id)
		if existing is None:
			copied = dict(chunk)
			copied["chunk_id"] = chunk_id
			copied.setdefault("id", chunk_id)
			copied["rrf_score"] = contribution
			merged[chunk_id] = copied
			return

		existing["rrf_score"] = float(existing.get("rrf_score", 0.0)) + contribution
		cls._merge_best_scores(existing, chunk)

	@staticmethod
	def _merge_chunk_lists(chunk_lists: list[list[dict[str, object]]]) -> list[dict[str, object]]:
		merged: dict[str, dict[str, object]] = {}
		for chunk_list in chunk_lists:
			for chunk in chunk_list:
				chunk_id = HybridRetriever._chunk_id(chunk)
				if not chunk_id:
					continue

				existing = merged.get(chunk_id)
				if existing is None:
					copied = dict(chunk)
					copied["chunk_id"] = chunk_id
					copied.setdefault("id", chunk_id)
					merged[chunk_id] = copied
					continue

				HybridRetriever._merge_best_scores(existing, chunk)

		results = list(merged.values())
		results.sort(key=lambda item: float(item.get("rerank_score", item.get("rrf_score", 0.0))), reverse=True)
		return results

	@staticmethod
	def _merge_best_scores(existing: dict[str, object], incoming: dict[str, object]) -> None:
		for field in ("similarity_score", "bm25_score", "rrf_score", "rerank_score", "distance"):
			incoming_value = incoming.get(field)
			if incoming_value is None:
				continue
			current_value = existing.get(field)
			if current_value is None or float(incoming_value) > float(current_value):
				existing[field] = float(incoming_value)

		if not existing.get("text") and incoming.get("text"):
			existing["text"] = incoming["text"]
		if not existing.get("metadata") and incoming.get("metadata"):
			existing["metadata"] = incoming["metadata"]

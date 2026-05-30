from __future__ import annotations

import logging
import threading
import time
from typing import ClassVar

import numpy as np
from sentence_transformers import CrossEncoder

from config import settings

logger = logging.getLogger(__name__)


class Reranker:
	_instance: ClassVar[Reranker | None] = None
	_initialized: ClassVar[bool] = False
	_singleton_lock: ClassVar[threading.Lock] = threading.Lock()

	def __new__(cls) -> Reranker:
		if cls._instance is None:
			with cls._singleton_lock:
				if cls._instance is None:
					cls._instance = super().__new__(cls)
		return cls._instance

	def __init__(self) -> None:
		if self.__class__._initialized:
			return

		self._model = CrossEncoder(settings.RERANKER_MODEL)
		self.__class__._initialized = True
		logger.info("Loaded reranker model: %s", settings.RERANKER_MODEL)

	@classmethod
	def is_loaded(cls) -> bool:
		return cls._initialized

	def rerank(self, query: str, chunks: list[dict], top_k: int) -> list[dict]:
		if not chunks or top_k <= 0:
			return []

		start_time = time.perf_counter()
		pairs = [(query, str(chunk.get("text", "") or "")) for chunk in chunks]
		scores = self._model.predict(pairs, show_progress_bar=False)
		scores_array = np.asarray(scores, dtype=np.float32).tolist()

		ranked_chunks = []
		for chunk, score in zip(chunks, scores_array):
			ranked_chunk = dict(chunk)
			ranked_chunk["rerank_score"] = float(score)
			ranked_chunks.append(ranked_chunk)

		ranked_chunks.sort(key=lambda item: float(item.get("rerank_score", 0.0)), reverse=True)
		elapsed_ms = (time.perf_counter() - start_time) * 1000.0
		logger.info("Reranked %s chunks in %.2f ms", len(chunks), elapsed_ms)
		return ranked_chunks[:top_k]

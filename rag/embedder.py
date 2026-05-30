from __future__ import annotations

import logging
import threading
from typing import ClassVar

import numpy as np
from sentence_transformers import SentenceTransformer

from config import settings

logger = logging.getLogger(__name__)


class Embedder:
	_instance: ClassVar[Embedder | None] = None
	_initialized: ClassVar[bool] = False
	_singleton_lock: ClassVar[threading.Lock] = threading.Lock()
	_query_prefix: ClassVar[str] = "Represent this sentence for searching relevant passages: "
	_batch_size: ClassVar[int] = 32

	def __new__(cls) -> Embedder:
		if cls._instance is None:
			with cls._singleton_lock:
				if cls._instance is None:
					cls._instance = super().__new__(cls)
		return cls._instance

	def __init__(self) -> None:
		if self.__class__._initialized:
			return

		self._model = SentenceTransformer(settings.EMBEDDING_MODEL)
		self.__class__._initialized = True
		logger.info("Loaded embedding model: %s", settings.EMBEDDING_MODEL)

	@classmethod
	def is_loaded(cls) -> bool:
		return cls._initialized

	def embed_text(self, text: str) -> list[float]:
		logger.info("Embedding single text")
		return self._embed_single(f"{self._query_prefix}{text}")

	def embed_query(self, text: str) -> list[float]:
		logger.info("Embedding query")
		return self._embed_single(f"{self._query_prefix}{text}")

	def embed_batch(self, texts: list[str]) -> list[list[float]]:
		logger.info("Embedding batch of %s texts", len(texts))
		if not texts:
			return []

		embeddings = self._model.encode(
			texts,
			batch_size=self._batch_size,
			convert_to_numpy=True,
			normalize_embeddings=True,
			show_progress_bar=False,
		)
		normalized = np.asarray(embeddings, dtype=np.float32)
		results = normalized.tolist()
		logger.info("Embedded batch of %s texts", len(results))
		return results

	def _embed_single(self, text: str) -> list[float]:
		embedding = self._model.encode(
			text,
			convert_to_numpy=True,
			normalize_embeddings=True,
			show_progress_bar=False,
		)
		result = np.asarray(embedding, dtype=np.float32).tolist()
		logger.info("Generated embedding with %s dimensions", len(result))
		return result

	@staticmethod
	def normalize(vector: list[float]) -> list[float]:
		logger.info("Normalizing vector with %s dimensions", len(vector))
		array = np.asarray(vector, dtype=np.float32)
		norm = float(np.linalg.norm(array))
		if norm == 0.0:
			return array.tolist()
		return (array / norm).tolist()

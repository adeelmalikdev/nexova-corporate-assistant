from __future__ import annotations

import asyncio
import contextvars
import logging
from typing import Any

import httpx

from config import settings
from rag.retriever import HybridRetriever

logger = logging.getLogger(__name__)

_GROK_USAGE: contextvars.ContextVar[dict[str, int] | None] = contextvars.ContextVar(
	"grok_usage",
	default=None,
)


def _infer_domain(system_prompt: str) -> str:
	prompt = system_prompt.lower()
	if "hr policy assistant" in prompt:
		return "hr"
	if "legal policy assistant" in prompt:
		return "legal"
	if "finance policy assistant" in prompt:
		return "finance"
	if "engineering knowledge assistant" in prompt:
		return "engineering"
	return "unknown"


def _safe_float(value: Any, default: float = 0.0) -> float:
	try:
		if value is None:
			return default
		return float(value)
	except (TypeError, ValueError):
		return default


def _extract_usage(response_data: dict[str, Any]) -> dict[str, int] | None:
	usage = response_data.get("usage")
	if not isinstance(usage, dict):
		return None

	extracted: dict[str, int] = {}
	for key in ("prompt_tokens", "completion_tokens", "response_tokens", "total_tokens"):
		value = usage.get(key)
		if value is None:
			continue
		try:
			extracted[key] = int(value)
		except (TypeError, ValueError):
			continue

	return extracted or None


def _extract_response_text(response_data: dict[str, Any]) -> str:
	content = response_data.get("content")
	if isinstance(content, list) and content:
		first_item = content[0]
		if isinstance(first_item, dict):
			text = first_item.get("text")
			if text is not None:
				return str(text).strip()

	choices = response_data.get("choices")
	if isinstance(choices, list) and choices:
		first_choice = choices[0]
		if isinstance(first_choice, dict):
			message = first_choice.get("message")
			if isinstance(message, dict):
				text = message.get("content")
				if text is not None:
					return str(text).strip()

	raise ValueError("Unexpected Grok response format")


def _normalize_sources(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
	sources: list[dict[str, Any]] = []
	for chunk in chunks:
		metadata = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
		source_file = chunk.get("source_file") or metadata.get("source_file") or "unknown"
		page_number = chunk.get("page_number")
		if page_number is None:
			page_number = metadata.get("page_number", "unknown")

		sources.append(
			{
				"source_file": str(source_file),
				"page_number": page_number,
				"similarity_score": _safe_float(chunk.get("similarity_score")),
				"rerank_score": _safe_float(chunk.get("rerank_score")),
			}
		)

	return sources


def _compute_confidence(chunks: list[dict[str, Any]]) -> float:
	rerank_scores = [
		_safe_float(chunk.get("rerank_score"))
		for chunk in chunks
		if chunk.get("rerank_score") is not None
	]
	if rerank_scores:
		return sum(rerank_scores) / len(rerank_scores)

	similarity_scores = [
		_safe_float(chunk.get("similarity_score"))
		for chunk in chunks
		if chunk.get("similarity_score") is not None
	]
	if similarity_scores:
		return sum(similarity_scores) / len(similarity_scores)

	return 0.0


async def call_grok(system_prompt: str, user_message: str, max_tokens: int = 1000) -> str:
	domain = _infer_domain(system_prompt)
	if settings.DEV_MODE:
		mock_response = f"DEV MODE: [{domain}] agent response for: {user_message[:50]}"
		_GROK_USAGE.set({"prompt_tokens": 0, "response_tokens": 0})
		logger.info("Grok mock response | domain=%s", domain)
		return mock_response

	payload = {
		"model": settings.GROK_MODEL,
		"messages": [
			{"role": "system", "content": system_prompt},
			{"role": "user", "content": user_message},
		],
		"max_tokens": max_tokens,
		"temperature": 0,
	}

	headers = {
		"x-api-key": settings.GROK_API_KEY,
		"content-type": "application/json",
	}

	last_error: Exception | None = None
	for attempt in range(3):
		try:
			async with httpx.AsyncClient(
				base_url=settings.GROK_BASE_URL,
				headers=headers,
				timeout=httpx.Timeout(30.0),
			) as client:
				response = await client.post("/chat/completions", json=payload)

			if response.status_code in {429, 500}:
				raise httpx.HTTPStatusError(
					f"Grok request failed with status {response.status_code}",
					request=response.request,
					response=response,
				)

			response.raise_for_status()
			response_data = response.json()
			usage = _extract_usage(response_data)
			_GROK_USAGE.set(usage)
			text = _extract_response_text(response_data)
			prompt_tokens = usage.get("prompt_tokens") if usage else None
			response_tokens = usage.get("completion_tokens") if usage else None
			if response_tokens is None and usage is not None:
				response_tokens = usage.get("response_tokens")
			logger.info(
				"Grok response | domain=%s | prompt_tokens=%s | response_tokens=%s",
				domain,
				prompt_tokens,
				response_tokens,
			)
			return text
		except (httpx.HTTPStatusError, httpx.TimeoutException, httpx.RequestError) as exc:
			last_error = exc
			status_code = (
				exc.response.status_code
				if isinstance(exc, httpx.HTTPStatusError) and exc.response is not None
				else None
			)
			if attempt < 2 and status_code in {429, 500}:
				await asyncio.sleep(1)
				continue
			if attempt < 2 and status_code is None:
				raise
			break

	if last_error is not None:
		raise last_error
	raise RuntimeError("Grok request failed without a response")


class DomainAgent:
	def __init__(self, domain: str, system_prompt: str) -> None:
		self.domain = domain
		self.system_prompt = system_prompt

	async def answer(self, query: str, retrieved_chunks: list[dict[str, Any]]) -> dict[str, Any]:
		if not retrieved_chunks:
			return {
				"domain": self.domain,
				"answer": (
					f"No relevant documents found for this query in the {self.domain} knowledge base."
				),
				"sources": [],
				"confidence": 0.0,
				"chunk_count": 0,
				"has_content": False,
			}

		context = HybridRetriever.format_context(retrieved_chunks)
		user_message = (
			"Use the following retrieved context to answer the question. "
			"Stay strictly within the provided documents.\n\n"
			f"Context:\n{context}\n\n"
			f"Question: {query}"
		)

		answer_text = await call_grok(self.system_prompt, user_message)
		usage = _GROK_USAGE.get() or {}
		logger.info(
			"Domain agent response | domain=%s | prompt_tokens=%s | response_tokens=%s",
			self.domain,
			usage.get("prompt_tokens"),
			usage.get("completion_tokens", usage.get("response_tokens")),
		)

		return {
			"domain": self.domain,
			"answer": answer_text,
			"sources": _normalize_sources(retrieved_chunks),
			"confidence": _compute_confidence(retrieved_chunks),
			"chunk_count": len(retrieved_chunks),
			"has_content": True,
		}


HR_AGENT = DomainAgent(
	domain="hr",
	system_prompt="""You are the HR policy assistant for Nexova
Technologies Pvt. Ltd. Answer questions strictly based on the
provided HR documents. Rules:
1. Cite the specific document name and section for every claim.
2. If the answer is not in the provided documents, say explicitly:
   'This information was not found in the HR documents.'
3. Never speculate or add information from general knowledge.
4. If numbers or dates appear, quote them exactly as written.""",
)

LEGAL_AGENT = DomainAgent(
	domain="legal",
	system_prompt="""You are the legal policy assistant for Nexova
Technologies Pvt. Ltd. You provide information from company legal
documents only. Rules:
1. You do not give legal advice — you cite policy and document text.
2. Always include document name and clause/article number.
3. If a query asks for legal interpretation rather than policy
   lookup, flag it: 'This requires legal interpretation beyond
   policy lookup.'
4. Never speculate beyond the documents provided.""",
)

FINANCE_AGENT = DomainAgent(
	domain="finance",
	system_prompt="""You are the finance policy assistant for Nexova
Technologies Pvt. Ltd. Rules:
1. Answer based strictly on financial reports and expense policies.
2. Always cite figures with source document name and date.
3. If you see numbers that appear inconsistent, flag them explicitly:
   'Note: inconsistent figures detected across documents.'
4. Never estimate or extrapolate financial figures.""",
)

ENGINEERING_AGENT = DomainAgent(
	domain="engineering",
	system_prompt="""You are the engineering knowledge assistant for
Nexova Technologies Pvt. Ltd. Rules:
1. Answer based on architecture documents, runbooks, and technical
   references provided.
2. Be precise with technical details — do not paraphrase commands
   or configuration values.
3. If a runbook step is ambiguous, say so explicitly rather than
   interpreting it.
4. For incident runbook queries, always include the full step
   sequence, not a summary.""",
)

AGENTS = {
	"hr": HR_AGENT,
	"legal": LEGAL_AGENT,
	"finance": FINANCE_AGENT,
	"engineering": ENGINEERING_AGENT,
}


__all__ = [
	"AGENTS",
	"DomainAgent",
	"ENGINEERING_AGENT",
	"FINANCE_AGENT",
	"HR_AGENT",
	"LEGAL_AGENT",
	"call_grok",
]

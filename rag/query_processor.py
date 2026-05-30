from __future__ import annotations

import json
import logging

import httpx

from config import settings

logger = logging.getLogger(__name__)


class QueryProcessor:
	async def rewrite_query(self, query: str) -> str:
		"""Rewrite a user question into a search-optimized query."""
		system_prompt = (
			"Rewrite this user question as a keyword-rich search query "
			"optimized for retrieving relevant corporate policy and "
			"technical documents. Return only the rewritten query string, "
			"nothing else. No explanation."
		)

		try:
			rewritten = await self._call_grok(system_prompt, query)
			logger.info("Query rewrite | original=%r | rewritten=%r", query, rewritten)
			return rewritten or query
		except Exception:
			logger.exception("Failed to rewrite query; returning original query")
			return query

	async def expand_query(self, query: str) -> list[str]:
		"""Generate semantically different retrieval variants for a query."""
		system_prompt = (
			"Generate 3 semantically different versions of this query "
			"for document retrieval. Each version should emphasize "
			"different aspects. Return a JSON array of 3 strings only. "
			"No explanation, no markdown."
		)

		try:
			response_text = await self._call_grok(system_prompt, query)
			variants = self._parse_query_variants(response_text)
			if not variants:
				return [query]
			return self._deduplicate([variant for variant in variants if variant])
		except Exception:
			logger.exception("Failed to expand query; returning original query")
			return [query]

	async def process(self, query: str) -> dict[str, object]:
		"""Rewrite and expand a query for downstream retrieval."""
		rewritten = await self.rewrite_query(query)
		expanded_variants = await self.expand_query(rewritten)
		expanded = self._deduplicate([rewritten, *expanded_variants])
		return {
			"original": query,
			"rewritten": rewritten,
			"expanded": expanded,
		}

	async def _call_grok(self, system_prompt: str, user_prompt: str) -> str:
		payload = {
			"model": settings.GROK_MODEL,
			"messages": [
				{"role": "system", "content": system_prompt},
				{"role": "user", "content": user_prompt},
			],
			"temperature": 0,
		}

		headers = {
			"Authorization": f"Bearer {settings.GROK_API_KEY}",
			"Content-Type": "application/json",
		}

		async with httpx.AsyncClient(
			base_url=settings.GROK_BASE_URL,
			headers=headers,
			timeout=httpx.Timeout(30.0, connect=10.0),
		) as client:
			response = await client.post("/chat/completions", json=payload)
			response.raise_for_status()
			data = response.json()

		try:
			content = data["choices"][0]["message"]["content"]
		except (KeyError, IndexError, TypeError) as exc:
			raise ValueError("Unexpected Grok response format") from exc

		return self._clean_text_response(str(content))

	@staticmethod
	def _clean_text_response(text: str) -> str:
		cleaned = text.strip()
		if cleaned.startswith("```"):
			lines = [line.strip() for line in cleaned.splitlines()]
			lines = [line for line in lines if line not in {"```", "```json", "```text"}]
			cleaned = " ".join(lines).strip()
		if "\n" in cleaned:
			cleaned = cleaned.splitlines()[0].strip()
		return cleaned.strip("\"'` ")

	@staticmethod
	def _parse_query_variants(response_text: str) -> list[str]:
		text = response_text.strip()
		if text.startswith("```"):
			lines = [line.strip() for line in text.splitlines()]
			lines = [line for line in lines if line not in {"```", "```json", "```text"}]
			text = "\n".join(lines).strip()

		start = text.find("[")
		end = text.rfind("]")
		if start == -1 or end == -1 or end <= start:
			raise ValueError("Grok expansion response did not contain a JSON array")

		payload = json.loads(text[start : end + 1])
		if not isinstance(payload, list):
			raise ValueError("Grok expansion response was not a JSON array")

		variants = [str(item).strip() for item in payload if str(item).strip()]
		return QueryProcessor._deduplicate(variants)

	@staticmethod
	def _deduplicate(items: list[str]) -> list[str]:
		seen: set[str] = set()
		unique_items: list[str] = []
		for item in items:
			candidate = item.strip()
			if not candidate or candidate in seen:
				continue
			seen.add(candidate)
			unique_items.append(candidate)
		return unique_items

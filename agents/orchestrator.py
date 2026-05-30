from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from agents.domain_agents import AGENTS, call_grok
from agents.synthesis import SynthesisAgent
from config import settings
from rag.query_processor import QueryProcessor
from rag.retriever import HybridRetriever

logger = logging.getLogger(__name__)

VALID_DOMAINS = ["hr", "legal", "finance", "engineering"]
DEPARTMENT_NAMES_BY_ID = {
	1: "hr",
	2: "legal",
	3: "finance",
	4: "engineering",
}


def _clean_json_text(text: str) -> str:
	cleaned = text.strip()
	if cleaned.startswith("```"):
		lines = [line.strip() for line in cleaned.splitlines()]
		lines = [line for line in lines if line not in {"```", "```json", "```text"}]
		cleaned = "\n".join(lines).strip()
	return cleaned


def _safe_json_array(response_text: str) -> list[str]:
	text = _clean_json_text(response_text)
	start = text.find("[")
	end = text.rfind("]")
	if start == -1 or end == -1 or end <= start:
		raise ValueError("JSON array not found")

	payload = json.loads(text[start : end + 1])
	if not isinstance(payload, list):
		raise ValueError("Response was not a JSON array")

	validated: list[str] = []
	for item in payload:
		candidate = str(item).strip().lower()
		if candidate in VALID_DOMAINS and candidate not in validated:
			validated.append(candidate)

	return validated


class Orchestrator:
	def __init__(self) -> None:
		self._query_processor = QueryProcessor()
		self._retriever = HybridRetriever()
		self._synthesis_agent = SynthesisAgent()

	async def classify_query(self, query: str) -> list[str]:
		system_prompt = (
			"You are a query router for a corporate knowledge system with 4 domains: hr, legal, finance, engineering.\n"
			"- hr: employee policies, PTO, hiring, onboarding, conduct, compensation, benefits\n"
			"- legal: contracts, NDAs, compliance, GDPR, regulatory\n"
			"- finance: expenses, budgets, reimbursements, financial reports\n"
			"- engineering: architecture, CI/CD, incidents, runbooks, APIs\n"
			"Given a user query, return ONLY a valid JSON array of relevant domain names. Minimum 1 domain. Example: [\"hr\", \"finance\"].\n"
			"If unclear, return [\"hr\"] as default."
		)

		try:
			response_text = await call_grok(system_prompt, query, max_tokens=64)
			domains = _safe_json_array(response_text)
			if not domains:
				raise ValueError("No valid domains returned")
			logger.info("Query classified | query=%r | domains=%s", query, domains)
			return domains
		except Exception:
			logger.exception("Failed to classify query; falling back to hr")
			return ["hr"]

	def apply_permission_filter(
		self,
		domains: list[str],
		allowed_domains: list[str],
		department_id: int,
		role: str,
	) -> list[str]:
		permitted = {domain for domain in allowed_domains if domain in VALID_DOMAINS}

		if role in {"manager", "dept_head"}:
			department_domain = DEPARTMENT_NAMES_BY_ID.get(department_id)
			if department_domain in VALID_DOMAINS:
				permitted.add(department_domain)

		filtered = [domain for domain in domains if domain in permitted]
		logger.info(
			"Permission filter applied | role=%s | department_id=%s | input=%s | permitted=%s | output=%s",
			role,
			department_id,
			domains,
			sorted(permitted),
			filtered,
		)
		return filtered

	async def route(
		self,
		query: str,
		allowed_domains: list[str],
		department_id: int,
		role: str,
	) -> dict[str, Any]:
		processed = await self._query_processor.process(query)
		original_query = str(processed["original"])
		rewritten_query = str(processed["rewritten"])
		expanded_queries = list(processed["expanded"])

		classified_domains = await self.classify_query(rewritten_query)
		permitted_domains = self.apply_permission_filter(
			classified_domains,
			allowed_domains,
			department_id,
			role,
		)

		if not permitted_domains:
			return {
				"answer": (
					"You don't have permission to access the domains relevant to this query. "
					"Contact your administrator."
				),
				"domains_consulted": [],
				"permission_denied": True,
				"requires_human_review": False,
			}

		async def retrieve_and_answer(domain: str) -> tuple[str, dict[str, Any]]:
			retrieved = await self._retriever.retrieve_multi_domain(
				expanded_queries,
				[domain],
				settings.TOP_K_RERANKED,
			)
			chunks = retrieved.get(domain, [])
			agent = AGENTS[domain]
			response = await agent.answer(original_query, chunks)
			return domain, response

		domain_tasks = [retrieve_and_answer(domain) for domain in permitted_domains]
		domain_results = await asyncio.gather(*domain_tasks, return_exceptions=True)

		responses: dict[str, dict[str, Any]] = {}
		for domain, result in zip(permitted_domains, domain_results):
			if isinstance(result, Exception):
				logger.exception("Domain routing failed for domain=%s", domain)
				responses[domain] = {
					"domain": domain,
					"answer": "",
					"sources": [],
					"confidence": 0.0,
					"chunk_count": 0,
					"has_content": False,
				}
			else:
				result_domain, response = result
				responses[result_domain] = response

		contentful_responses = [
			response for response in responses.values() if bool(response.get("has_content"))
		]

		if len(permitted_domains) == 1 and len(contentful_responses) == 1:
			single_domain = permitted_domains[0]
			single_response = dict(responses[single_domain])
			single_response.update(
				{
					"domains_consulted": [single_domain],
					"query_variants": expanded_queries,
					"domains_classified": classified_domains,
					"permission_denied": False,
					"requires_human_review": False,
				}
			)
			return single_response

		synthesis_response = await self._synthesis_agent.synthesize(
			original_query,
			responses,
		)
		synthesis_response.update(
			{
				"query_variants": expanded_queries,
				"domains_classified": classified_domains,
				"permission_denied": False,
			}
		)
		return synthesis_response


__all__ = ["Orchestrator", "VALID_DOMAINS"]

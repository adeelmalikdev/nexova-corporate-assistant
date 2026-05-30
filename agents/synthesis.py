from __future__ import annotations

import logging
from typing import Any

from agents.domain_agents import call_grok

logger = logging.getLogger(__name__)


def _extract_conflicts(response_text: str) -> list[str]:
	text = response_text.strip()
	upper_text = text.upper()
	marker = "CONFLICTS:"
	marker_index = upper_text.find(marker)
	if marker_index == -1:
		return []

	conflict_block = text[marker_index + len(marker) :].strip()
	if not conflict_block:
		return []

	conflicts = []
	for line in conflict_block.splitlines():
		candidate = line.strip().lstrip("-•*").strip()
		if not candidate or candidate.lower() == "none detected":
			continue
		conflicts.append(candidate)
	return conflicts


def _extract_answer(response_text: str) -> str:
	text = response_text.strip()
	upper_text = text.upper()
	answer_marker = "ANSWER:"
	conflicts_marker = "CONFLICTS:"
	answer_index = upper_text.find(answer_marker)
	if answer_index == -1:
		return text

	start = answer_index + len(answer_marker)
	conflicts_index = upper_text.find(conflicts_marker, start)
	if conflicts_index == -1:
		return text[start:].strip()

	return text[start:conflicts_index].strip()


class SynthesisAgent:
	async def synthesize(self, query: str, domain_responses: dict[str, dict[str, Any]]) -> dict[str, Any]:
		contentful_responses = {
			domain: response
			for domain, response in domain_responses.items()
			if bool(response.get("has_content"))
		}

		if not contentful_responses:
			return {
				"answer": "No relevant information found across any accessible knowledge base for this query.",
				"domains_consulted": list(domain_responses.keys()),
				"conflicts": [],
				"confidence": 0.0,
				"requires_human_review": False,
			}

		sections: list[str] = []
		all_sources: list[dict[str, Any]] = []
		confidences: list[float] = []
		for domain, response in contentful_responses.items():
			confidence = float(response.get("confidence", 0.0) or 0.0)
			confidences.append(confidence)
			sources = response.get("sources") if isinstance(response.get("sources"), list) else []
			all_sources.extend([source for source in sources if isinstance(source, dict)])
			sections.append(
				f"DOMAIN: {domain} (confidence: {confidence:.0%})\n"
				f"ANSWER: {response.get('answer', '')}\n"
				f"SOURCES: {sources}"
			)

		user_message = (
			f"User query: {query}\n\n"
			"Domain responses:\n"
			+ "\n\n".join(sections)
			+ "\n\n"
			"Synthesize these answers into one response. Follow the required structure exactly."
		)

		try:
			response_text = await call_grok(
				"You are a synthesis agent for Nexova Technologies' corporate knowledge system. You receive answers from multiple domain-specific agents and must produce a unified response.\n"
				"Rules:\n"
				"1. Attribute each point to its source domain explicitly.\n"
				"2. If answers from different domains contradict each other, flag it in a CONFLICTS section — never silently pick one.\n"
				"3. Reflect uncertainty where domain confidence is low.\n"
				"4. Structure your response as:\n"
				"   ANSWER: [unified answer with domain attributions]\n"
				"   CONFLICTS: [list any contradictions, or 'None detected']\n"
				"5. Be concise — don't repeat the same point for each domain.",
				user_message,
				max_tokens=900,
			)
		except Exception:
			logger.exception("Synthesis Grok call failed; returning a fallback response")
			response_text = "ANSWER: Unable to synthesize a response at this time.\nCONFLICTS: None detected"

		answer_text = _extract_answer(response_text)

		conflicts = _extract_conflicts(response_text)
		average_confidence = sum(confidences) / len(confidences) if confidences else 0.0

		return {
			"answer": answer_text,
			"domains_consulted": list(contentful_responses.keys()),
			"sources": all_sources,
			"conflicts": conflicts,
			"confidence": average_confidence,
			"requires_human_review": len(conflicts) > 0,
			"query_variants": [],
		}


__all__ = ["SynthesisAgent"]

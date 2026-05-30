from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field, field_validator

from agents.orchestrator import VALID_DOMAINS
from auth.dependencies import require_password_changed
from config import settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Query"])

DOMAIN_DISPLAY_NAMES: dict[str, str] = {
	"hr": "Human Resources",
	"legal": "Legal & Compliance",
	"finance": "Finance",
	"engineering": "Engineering",
}


class QueryRequest(BaseModel):
	query: str = Field(...)

	@field_validator("query")
	@classmethod
	def _validate_query(cls, value: str) -> str:
		cleaned = value.strip()
		if not cleaned:
			raise ValueError("query is required")
		if len(cleaned) > settings.MAX_QUERY_LENGTH:
			raise ValueError(f"query must be at most {settings.MAX_QUERY_LENGTH} characters")
		return cleaned


class DomainStatsResponse(BaseModel):
	domain: str
	display_name: str
	chunk_count: int
	last_ingested: str | None


def _get_orchestrator(request: Request):
	orchestrator = getattr(request.app.state, "orchestrator", None)
	if orchestrator is None:
		raise HTTPException(
			status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
			detail="Query orchestrator is not available",
		)
	return orchestrator


def _get_chroma_client(request: Request):
	chroma_client = getattr(request.app.state, "chroma_client", None)
	if chroma_client is None:
		raise HTTPException(
			status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
			detail="Vector store is not available",
		)
	return chroma_client


def _domain_summary(chroma_client, domain: str) -> DomainStatsResponse:
	stats = chroma_client.get_collection_stats(domain)
	return DomainStatsResponse(
		domain=domain,
		display_name=DOMAIN_DISPLAY_NAMES[domain],
		chunk_count=int(stats.get("chunk_count", 0)),
		last_ingested=stats.get("last_ingested"),
	)


@router.post("")
async def query_assistant(
	body: QueryRequest,
	request: Request,
	current_user: dict = Depends(require_password_changed),
) -> dict[str, Any]:
	orchestrator = _get_orchestrator(request)
	query_text = body.query.strip()
	if not query_text:
		raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="query is required")
	if len(query_text) > settings.MAX_QUERY_LENGTH:
		raise HTTPException(
			status_code=status.HTTP_400_BAD_REQUEST,
			detail=f"query must be at most {settings.MAX_QUERY_LENGTH} characters",
		)

	start_time = time.perf_counter()
	role = current_user.get("role") or current_user.get("role_name") or ""
	response = await orchestrator.route(
		query=query_text,
		allowed_domains=current_user["allowed_domains"],
		department_id=current_user["department_id"],
		role=role,
	)
	elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
	logger.info(
		"Query handled | username=%s | query=%r | domains_consulted=%s | response_time_ms=%s | confidence=%s",
		current_user["employee"].username,
		query_text[:100],
		response.get("domains_consulted", []),
		elapsed_ms,
		response.get("confidence"),
	)
	return response


@router.get("/domains", response_model=list[DomainStatsResponse])
async def query_domains(
	request: Request,
	current_user: dict = Depends(require_password_changed),
) -> list[DomainStatsResponse]:
	orchestrator = _get_orchestrator(request)
	chroma_client = _get_chroma_client(request)
	role = current_user.get("role") or current_user.get("role_name") or ""
	accessible_domains = orchestrator.apply_permission_filter(
		list(VALID_DOMAINS),
		current_user["allowed_domains"],
		current_user["department_id"],
		role,
	)
	return [_domain_summary(chroma_client, domain) for domain in accessible_domains]
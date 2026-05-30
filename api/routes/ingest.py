from __future__ import annotations

import logging
import uuid
from datetime import date, datetime
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from pydantic import BaseModel

from auth.dependencies import require_admin
from rag.ingestor import DocumentIngestor

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Document Ingestion"])

VALID_DOMAINS = ("hr", "legal", "finance", "engineering")
DOMAIN_DISPLAY_NAMES: dict[str, str] = {
	"hr": "Human Resources",
	"legal": "Legal & Compliance",
	"finance": "Finance",
	"engineering": "Engineering",
}


class IngestStatusResponse(BaseModel):
	domain: str
	display_name: str
	chunk_count: int
	last_ingested: str | None


def _get_ingestor(request: Request) -> DocumentIngestor:
	ingestor = getattr(request.app.state, "document_ingestor", None)
	if ingestor is None:
		raise HTTPException(
			status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
			detail="Document ingestor is not available",
		)
	return ingestor


def _get_chroma_client(request: Request):
	chroma_client = getattr(request.app.state, "chroma_client", None)
	if chroma_client is None:
		raise HTTPException(
			status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
			detail="Vector store is not available",
		)
	return chroma_client


def _normalize_domain(domain: str) -> str:
	normalized = domain.strip().lower()
	if normalized not in VALID_DOMAINS:
		raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid domain")
	return normalized


def _parse_effective_date(value: str) -> date:
	try:
		return datetime.strptime(value.strip(), "%Y-%m-%d").date()
	except ValueError as exc:
		raise HTTPException(
			status_code=status.HTTP_400_BAD_REQUEST,
			detail="effective_date must be in YYYY-MM-DD format",
		) from exc


def _domain_status(chroma_client, domain: str) -> IngestStatusResponse:
	stats = chroma_client.get_collection_stats(domain)
	return IngestStatusResponse(
		domain=domain,
		display_name=DOMAIN_DISPLAY_NAMES[domain],
		chunk_count=int(stats.get("chunk_count", 0)),
		last_ingested=stats.get("last_ingested"),
	)


@router.post("")
async def ingest_document(
	request: Request,
	file: UploadFile = File(...),
	domain: str = Form(...),
	version: str = Form(...),
	effective_date: str = Form(...),
	current_user: dict = Depends(require_admin),
) -> dict:
	ingestor = _get_ingestor(request)
	normalized_domain = _normalize_domain(domain)
	parsed_effective_date = _parse_effective_date(effective_date)
	if not file.filename:
		raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="file is required")

	temp_path = Path("/tmp") / f"{uuid.uuid4()}_{Path(file.filename).name}"
	metadata = {
		"version": version.strip(),
		"effective_date": parsed_effective_date,
	}

	try:
		content = await file.read()
		temp_path.write_bytes(content)
		result = await ingestor.ingest_pdf(str(temp_path), normalized_domain, metadata)
		logger.info(
			"Document ingested | username=%s | domain=%s | source_file=%s | chunks=%s",
			current_user["employee"].username,
			normalized_domain,
			result.get("source_file"),
			result.get("chunks_ingested"),
		)
		return result
	finally:
		temp_path.unlink(missing_ok=True)


@router.get("/status", response_model=list[IngestStatusResponse])
async def ingest_status(
	request: Request,
	current_user: dict = Depends(require_admin),
) -> list[IngestStatusResponse]:
	del current_user
	chroma_client = _get_chroma_client(request)
	return [_domain_status(chroma_client, domain) for domain in VALID_DOMAINS]


@router.get("/status/{domain}", response_model=IngestStatusResponse)
async def ingest_status_for_domain(
	domain: str,
	request: Request,
	current_user: dict = Depends(require_admin),
) -> IngestStatusResponse:
	del current_user
	chroma_client = _get_chroma_client(request)
	return _domain_status(chroma_client, _normalize_domain(domain))
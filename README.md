# Nexova Corporate Assistant

Nexova Corporate Assistant is a multi-domain FastAPI application that combines role-based authentication, employee lifecycle management, and a multi-agent RAG pipeline for internal knowledge search across HR, legal, finance, and engineering documents. It uses hybrid retrieval, domain-aware access control, and agent orchestration to answer corporate questions from ingested PDF sources while keeping each user constrained to the domains they are allowed to access.

## Architecture Overview

```text
User
	-> FastAPI API
		-> JWT Auth + Role/Department Policy
			-> Query Orchestrator
				-> Query Rewriter / Multi-Query Expansion
				-> Hybrid Retriever (BM25 + Dense)
				-> Cross-Encoder Reranker
				-> Domain Agents (HR, Legal, Finance, Engineering)
				-> Synthesis Agent
					-> Response

PDF Documents
	-> Ingest API
		-> Chunking + Metadata
			-> Sentence Transformers Embeddings
			-> ChromaDB Vector Store
			-> SQLite Metadata / Users / Roles
```

## Key Features

- Multi-agent RAG with an orchestrator, four domain agents, and a synthesis layer.
- Hybrid search that combines BM25 lexical scoring with dense semantic retrieval.
- Cross-encoder reranking to improve precision on the final candidate set.
- JWT authentication with role-based and department-aware access control.
- Employee registration, verification, role assignment, and account lifecycle workflows.
- Conflict detection across domain sources for safer internal answers.

## Tech Stack

| Component | Purpose |
| --- | --- |
| FastAPI | Async web API and dependency injection |
| SQLite | Lightweight relational storage for users, roles, and employee data |
| ChromaDB | Persistent vector store for ingested document chunks |
| sentence-transformers | Embeddings for semantic retrieval |
| Grok API | Query classification and synthesis support |
| JWT | Stateless authentication and authorization |

## Project Structure

```text
.
├── api/
│   ├── main.py
│   └── routes/
├── agents/
├── auth/
├── database/
├── documents/
│   ├── engineering/
│   ├── finance/
│   ├── hr/
│   └── legal/
├── rag/
├── vectordb/
├── config.py
├── seed.py
├── requirements.txt
└── pyproject.toml
```

## Setup

1. Clone the repository.
2. Create and activate the virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Create your local environment file from the example template:

```bash
cp .env.example .env
```

5. Fill in the required secrets and any environment-specific values in `.env`.

## Run the Server

```bash
source .venv/bin/activate
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://127.0.0.1:8000`, with interactive docs at `/docs`.

## Ingest Documents

The ingest endpoint accepts a PDF plus metadata for the target domain.

```bash
curl -X POST "http://127.0.0.1:8000/ingest" \
	-H "Authorization: Bearer <admin-jwt>" \
	-F "file=@./documents/hr/policy.pdf" \
	-F "domain=hr" \
	-F "version=1.0" \
	-F "effective_date=2026-05-30"
```

## Default Admin Credentials

The seed data creates an initial administrator account:

- Username: `admin`
- Email: `admin@nexova.io`
- Password: `Admin@nexova1`

Change the password immediately after the first login.

## Role to Domain Access

| Role | Base Allowed Domains | Notes |
| --- | --- | --- |
| employee | hr | Default employee access |
| manager | hr | Department-aware escalation may expand access at runtime |
| dept_head | hr, legal | Department-aware escalation may expand access at runtime |
| hr_admin | hr, finance | Used for employee administration |
| admin | hr, legal, finance, engineering | Full system access |

## Key API Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| POST | `/auth/login` | Authenticate and return a JWT |
| POST | `/auth/logout` | Invalidate the current session token |
| GET | `/auth/me` | Fetch the current user profile |
| POST | `/auth/change-password` | Change the active password |
| POST | `/employees/register` | Create a new employee account |
| GET | `/employees` | List employees |
| GET | `/employees/pending` | List unverified employees |
| PATCH | `/employees/{employee_id}/verify` | Verify a pending employee |
| PATCH | `/employees/{employee_id}/role` | Update an employee role |
| POST | `/ingest` | Upload and index a PDF document |
| GET | `/ingest/status` | View domain ingestion stats |
| GET | `/query/domains` | Show domains available to the current user |
| POST | `/query` | Ask the assistant a question |
| GET | `/health` | Check service health |

## Known Limitations

- The system is designed around local SQLite and ChromaDB persistence, so it is best suited for small to medium internal deployments or demos.
- Ingestion currently expects PDF files and domain metadata.
- The repository uses a permissive CORS policy in development and should be tightened before production.
- Grok API access requires a valid external API key and network connectivity.
- Initial document quality depends on the contents and structure of the PDFs in `documents/`.

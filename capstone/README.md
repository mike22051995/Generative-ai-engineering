# Document Q&A API

A production-grade Document Q&A system built from scratch using RAG (Retrieval Augmented Generation). No LangChain. No shortcuts. Pure engineering fundamentals.

Built by a Senior Backend Engineer at Nutanix to demonstrate that backend engineers are uniquely positioned to build production AI systems.

---

## Performance Benchmarks

| Metric | Value |
|--------|-------|
| Cache HIT | 64ms |
| Cache MISS (warm) | 4.7s |
| Cache speedup | 73x |
| Documents supported | Text + PDF |
| Vector dimensions | 1536 |
| Vector index | HNSW (cosine) |

---

## Architecture
User
│
▼
FastAPI ──► Redis Cache ──► HIT: return in 64ms
│ │
│ └─► MISS: continue pipeline
│
▼
OpenAI Embeddings (query)
│
▼
pgvector HNSW Search
│
▼
MMR Retrieval (diversity filter)
│
▼
Parallel LLM Reranking (ThreadPoolExecutor)
│
▼
LLM Answer Generation (grounded, no hallucination)
│
▼
Cache result → Return JSON with answer + sources

---

## Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| API | FastAPI | async, production-grade, auto docs |
| Vector DB | PostgreSQL + pgvector | no new infra, HNSW index |
| Cache | Redis | 73x speedup on repeated queries |
| Embeddings | OpenAI text-embedding-3-small | 1536 dims, cost-effective |
| LLM | GPT-3.5-turbo | fast, accurate, affordable |
| PDF | PyMuPDF | fastest PDF extraction library |
| Config | Pydantic Settings | type-safe, fails fast at startup |
| Infra | Docker Compose | one command runs everything |

---

## Quick Start

Prerequisites: Docker, OpenAI API key

```bash
git clone https://github.com/mike22051995/Generative-ai-engineering.git
cd generative-ai-engineering/capstone
cp .env.example .env
# Add your OPENAI_API_KEY to .env
docker compose up --build
```

API: http://localhost:8000

Docs: http://localhost:8000/docs

---

## API Reference

### Upload Text Document

```bash
POST /documents
Content-Type: application/json

{
    "text": "document content",
    "filename": "policy.txt"
}
```

### Upload PDF

```bash
POST /documents/pdf
Content-Type: multipart/form-data

file: document.pdf
```

### Query

```bash
POST /query
Content-Type: application/json

{
    "question": "What is the refund policy?",
    "use_cache": true
}
```

### Response

```json
{
    "answer": "Customers can return products within 30 days...",
    "sources": [{"id": 1, "text": "...", "score": 9.0}],
    "chunks_used": 3,
    "tokens_used": 344,
    "cached": false
}
```

---

## Key Engineering Decisions

**Connection pooling** — 10 reusable PostgreSQL connections, not one per request

**Batch embedding** — one OpenAI API call for all chunks, not N sequential calls

**Parallel reranking** — ThreadPoolExecutor scores chunks simultaneously, not sequentially

**ON CONFLICT DO NOTHING** — safe re-indexing, no duplicates ever

**Cache invalidation on upload** — new document clears stale cached answers

**RealDictCursor** — dict access over tuple access, schema changes never break the app

**init.sql** — tables created automatically on first Docker start, zero manual setup

---

## Project Structure
capstone/
├── main.py # FastAPI routes
├── config.py # Pydantic settings
├── database.py # Connection pool
├── cache.py # Redis cache layer
├── embeddings.py # OpenAI single + batch embedding
├── chunker.py # Recursive chunking with overlap
├── retriever.py # MMR + parallel reranking
├── generator.py # Grounded LLM generation
├── ingestor.py # Full ingestion pipeline
├── pdf_extractor.py # PyMuPDF text extraction
├── models.py # Pydantic request/response models
├── init.sql # Schema — auto-runs on first start
├── Dockerfile # Container definition
├── docker-compose.yml # Full stack orchestration
└── requirements.txt # Dependencies

---

## Author

Mukesh Prasad — Senior Backend Engineer at Nutanix

Part of a 7-module GenAI engineering journey building LLM and RAG systems from scratch.

Full repo: https://github.com/mike22051995/Generative-ai-engineering
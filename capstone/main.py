from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from models import (
    DocumentsUploadRequest,
    DocumentUploadResponse,
    QueryRequest,
    QueryResponse,
    ChunkSource,
    DocumentInfo
)
from ingestor import ingest_document
from retriever import retrieve
from generator import generate_answer
from cache import get_cached_answer, cache_answer
from database import execute_query
from config import settings

#--------------------------------APP STARTUP---------------------------------------------
@asynccontextmanager
async def lifespan(app:FastAPI):
    "Runs on startup and shutdown"
    print("starting Document Q & A API")
    print(f"DB: {settings.DB_HOST}: {settings.DB_PORT}/{settings.DB_NAME}")
    print(f"Redis :{settings.REDIS_HOST}: {settings.REDIS_PORT}")
    yield
    print("Shutting down.....")

app=FastAPI(
    title="Document Q&A API",
    description="Production RAG pipeline - upload docs, ask questions",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
def health_check():
    """Quick check that application is running."""
    return {"status":"healthy", "version":"1.0.0"}


#-----------------------------DOCUMENTS ENDPOINT------------------------------------------------

@app.post("/documents", response_model=DocumentUploadResponse)
def upload_document(request:DocumentsUploadRequest):
    """
    Upload and index a document.
    Chunks, embeds, and stores in pgvector.
    Invalidates Redis cache."""
    try:
        result=ingest_document(
            text=request.text,
            filename=request.filename,
            metadata=request.metadata
        )
        if result["status"]=="skipped":
            return DocumentUploadResponse(
                status="skipped",
                filename=request.filename,
                chunks_indexed=0,
                total_chunks=0,
                message=result["reason"]
            )
        return DocumentUploadResponse(
            status="success",
            filename=request.filename,
            chunks_indexed=result["chunks_indexed"],
            total_chunks=result["total_chunks"],
            message=f"Successfully indexed {result["chunks_indexed"]} chunks"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents")
def list_documents():
    """
    List all indexed documents.
    Returns one row per chunk with preview.
    """
    try:
        results=execute_query("""
            select id, filename, chunk_index,total_chunks, created_at,
            LEFT(content,100) as content_preview
            FROM DOCUMENTS
            ORDER BY filename, chunk_index""")
        return {
            "total_chunks":len(results),
            "documents":[
                {
                    "id":row["id"],
                    "filename":row["filename"],
                    "chunk_index":row["chunk_index"],
                    "total_chunks":row["total_chunks"],
                    "created_at":str(row["created_at"]),
                    "content_preview":row["content_preview"]
                }
                for row in results
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

#----------------------------------QUERY ENDPOINTS------------------------------------

@app.post("/query", response_model=QueryResponse)
def query_documents(request:QueryRequest): 
    """
    Ask a question about indexed documents.
    Uses Redis cache, pgvector retrieval , and LLM generation.
    """
    try:
        #step 1 - Check cache
        if request.use_cache:
            cached=get_cached_answer(request.question)
            if cached:
                return QueryResponse(
                    question=request.question,
                    answer=cached["answer"],
                    sources=[
                        ChunkSource(**s) for s in cached["sources"]
                    ],
                    chunks_used=cached["chunks_used"],
                    tokens_used=cached.get("tokens_used",0),
                    cached=True
                )  
        #step 2 - Retrieve relevant chunks
        chunks=retrieve(request.question)
        if not chunks:
            return QueryResponse(
                question=request.question,
                answer="I don't have information about that.",
                sources=[],
                chunks_used=0,
                tokens_used=0,
                cached=False
            )

        #Step 3 - generate answer
        result=generate_answer(request.question, chunks)

        #step 4 - Cache the result
        if request.use_cache:
            cache_answer(request.question, result)
        return QueryResponse(
            question=request.question,
            answer=result["answer"],
            sources=[
                ChunkSource(**s) for s in result["sources"]
            ],
            chunks_used=result["chunks_used"],
            tokens_used=result.get("tokens_used",0),
            cached=False
        )   
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

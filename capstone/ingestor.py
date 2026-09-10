from database import execute_write,execute_query
from embedding import get_embeddings_batch
from chunker import chunk_document
from cache import invalidate_cache
from config import settings

def document_exists(filename:str)->bool:
    """
    Check if document with this filename already indexed.
    Prevents duplicate indexing
    """

    results=execute_query("SELECT id FROM documents WHERE filename=%s LIMIT 1",(filename,))
    return len(results)>0


def ingest_document(text:str,filename:str="unknown", metadata:dict=None)->dict:
    """
    Full document ingestion pipeline:
    1. Check for duplicates
    2. Chunk document
    3. Embed all chunks in batch
    4. Store in PostgrSQL
    5. Invalidate cache
    Returns summary of what was indexed.
    """
    #Step 1- Check duplicate
    if document_exists(filename):
        return {
            "status":"skipped",
            "reason":f"document '{filename}' already indexed",
            "chunks_indexed":0
        }
    #Step 2 - Chunk document
    print(f"chunking document:{filename}")
    chunks=chunk_document(text, metadata={
        "filename":filename,
        **(metadata or {})
    })
    print(f"created {len(chunks)} chunks")
    if not chunks:
        return {
            "status":"error",
            "reason":"Document produced no chunks",
            "chunks_indexed":0
        }
    #Step 3 - batch indexed all chunks
    print(f"Embedding {len(chunks)} chunks.....")
    texts=[chunk["content"] for chunk in chunks]
    embeddings=get_embeddings_batch(texts)

    #Step 4 - Store in postgreSQL
    print("Storing chunks in postgreSQL")
    chunks_indexed=0
    for chunk, embedding in zip(chunks, embeddings):
        execute_write("""
            INSERT INTO documents(
            content,
            embedding,
            filename,
            chunk_index,
            total_chunks)
            VALUES (%s,%s,%s,%s,%s)
            ON CONFLICT DO NOTHING""",
            (
                chunk["content"],
                embedding,
                filename,
                chunk["chunk_index"],
                chunk["total_chunks"]
            ))
        chunks_indexed+=1

    #Step 5 - Invalidate cache
    deleted_keys=invalidate_cache()
    print(f"Cache invalidated: {deleted_keys} keys deleted")
    return {
        "status":"success",
        "filename":filename,
        "chunks_indexed":chunks_indexed,
        "total_chunks":len(chunks),
        "cache_keys_deleted":deleted_keys
    }
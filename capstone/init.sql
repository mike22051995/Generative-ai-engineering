-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create documents table
CREATE TABLE IF NOT EXISTS documents(
    id              SERIAL PRIMARY KEY,
    content         TEXT NOT NULL,
    embedding      vector(1536),
    filename        TEXT DEFAULT 'unknown',
    chunk_index     INTEGER DEFAULT 0,
    total_chunks    INTEGER DEFAULT 1,
    created_at      TIMESTAMP DEFAULT NOW()  
);

-- Create HNSW index for fast similarity search
CREATE INDEX IF NOT EXISTS documents_embedding_idx
ON documents
USING hnsw (embedding vector_cosine_ops);
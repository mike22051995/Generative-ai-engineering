from pydantic import BaseModel,Field 
from typing import Optional
from datetime import datetime

#REQUEST MODEL

class DocumentsUploadRequest(BaseModel):
    text:str=Field(
        ...,
        min_length=10,
        description="The document text to index"
    )
    filename:str=Field(
        default="unknown",
        description="Name of the document"

    )
    metadata:Optional[dict]=Field(
        default=None,
        description="Optional metadata about the document"
    )


class QueryRequest(BaseModel):
    question:str=Field(
        ...,
        min_length=3,
        description="The question to answer"
    )
    use_cache:bool=Field(
        default=True,
        description="Whether to use Redis cache"
    )

#RESPONSE_MODEL
#What the API returns to the user

class ChunkSource(BaseModel):
    id:Optional[int]=None
    text:str
    score:float

class QueryResponse(BaseModel):
    question:str
    answer:str
    sources:list[ChunkSource]
    chunks_used:int
    tokens_used:int
    cached:bool=False

class DocumentUploadResponse(BaseModel):
    status:str
    filename:str
    chunks_indexed:int
    total_chunks:int
    message:str

class DocumentInfo(BaseModel):
    id:int
    filename:str
    chunk_index:int
    total_chunks:int
    created_At:datetime
    content_preview:str
    

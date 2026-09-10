from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    OPENAI_API_KEY:str
    EMBEDDING_MODEL:str="text-embedding-3-small"
    LLM_MODEL:str="gpt-3.5-turbo"
    #postgre
    DB_HOST:str="localhost"
    DB_PORT:int=5433
    DB_NAME:str="ragdb"
    DB_USER:str
    DB_PASSWORD:str

    #Redis
    REDIS_HOST:str="localhost"
    REDIS_PORT:int=6379
    REDIS_TTL:int=3600


    #RAG SETTINGS
    CHUNK_SIZE:int=500
    CHUNK_OVERLAP:int=50
    TOP_K:int=3
    FETCH_K:int=10

    class Config:
        env_file=".env"
        env_file_encoding="utf-8"

settings=Settings()

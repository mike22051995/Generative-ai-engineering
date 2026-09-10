import redis 
import json 
import hashlib
from config import settings


#-----------------REDIS CONNECTION---------------------

redis_client=redis.Redis(
    host=settings.REDIS_HOST,
    port=settings.REDIS_PORT,
    decode_responses=True,
    protocol=2


)

def get_cache_key(query:str)->str:
    """Generate consistent cache key from query.
    Normalise->hash->prefix with namespace."""

    normalized=query.lower().strip()
    hash_value=hashlib.md5(normalized.encode()).hexdigest()
    return f"rag:{hash_value}"

def get_cached_answer(query:str)->str | None:
    '''check if answer exist in cache, if exist return or else return None'''
    key=get_cache_key(query)
    cached=redis_client.get(key)
    if cached:
        print(f"cache hit for the query: {query[:50]}....")
        return json.loads(cached)   #json.loads → converts string back to dict on retrieval
    print(f"cache miss for the query: {query[:50]}.......")
    return None


def cache_answer(query:str,answer:dict)->None:
    """
    Store answer in redis with ttl.
    """
    key=get_cache_key(query)
    redis_client.setex(
        name=key,
        time=settings.REDIS_TTL,
        value=json.dumps(answer)   #json.dumps → converts dict to string for storage
    )
    print(f"cached answer for query: {query[:50]}......")


def invalidate_cache()->int:
    """Delete all the RAG cache.
    Called when new documents are uploaded-
    old cached answer might be outdate.
    Returns number of keys deleted."""
    keys=redis_client.keys("rag:*")
    if keys:
        return redis_client.delete(*keys)
    return 0
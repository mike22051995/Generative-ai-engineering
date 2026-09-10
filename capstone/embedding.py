from openai import OpenAI
from config import settings

client=OpenAI(api_key=settings.OPENAI_API_KEY)


def get_embedding(query:str)->list[float]:
    """
    Get embedding for a single query.
    Used query embedding at search time"""
    response=client.embeddings.create(
        model=settings.EMBEDDING_MODEL,
        input=query
    )
    return response.data[0].embedding


def get_embeddings_batch(texts:list[str])->list[list[float]]:
    """
    Get multiple embeddings for multiple texts in one api call.
    Used for document indexing - much faster than one by one embedding.

    Openai support 2048 texts per batch.
    for larger batches we split in group of 
    """
    if not texts:
        return []
    all_embeddings=[]
    #process in a batch of 100
    batch_size=100
    for i in range(0,len(texts),batch_size):
        batch=texts[i:i+batch_size]
        response=client.embeddings.create(
            model=settings.EMBEDDING_MODEL,
            input=batch
        )
        batch_embedding=[item.embedding for item in response.data]
        all_embeddings.extend(batch_embedding)
        print(f"Embedded batch: {1//batch_size +1}"
              f"/{(len(texts)-1)/batch_size+1}"
              f"({len(batch)} texts)")
    return all_embeddings

from database import execute_query
from embedding import get_embedding
from openai import OpenAI
from config import settings



client=OpenAI(api_key=settings.OPENAI_API_KEY)

#Basic retrieval

def basic_retrieve(query:str,top_k:int=None)->list[dict]:
    """Retrieve top_k most relevant chunks using
    pgvector cosine similarity search."""
    top_k=top_k or settings.TOP_K
    query_embedding=get_embedding(query)
    results=execute_query("""
        WITH scored AS(
        SELECT id,
            content,
            embedding<=>%s::vector AS distance
        FROM documents
        ORDER BY distance 
        LIMIT %s
        )
        SELECT id,
            content,
            1-distance AS similarity
        FROM scored 
        ORDER BY similarity DESC
        """,(query_embedding,settings.TOP_K))
    return [
        {
            "id":row["id"],
            "text":row["content"],
            "score":row["similarity"]
        } for row in results
    ]

#RE-RANKING

def rerank(query:str,chunks:list[dict],top_k:int=None)->list[dict]:
    """
    Use LLM to rerank chunks by actual answer quality.
    More accurate than embedding similarity alone.
    """
    top_k=top_k or settings.TOP_K
    reranked=[]


    for chunk in chunks:
        response=client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=[
                {
                    "role":"system",
                    "content":"""you are relevance scorer.
                    Score how well the given text answers the query.
                    0=completely irrelevant
                    10=perfectly relevant
                    No explanatioin. just the number."""
                },
                {"role":"user",
                 "content":f"Query: {query}\n\n Text :{chunk['text']}"
                 }
            ],
                temperature=0.0,
                max_tokens=5,
                
        )
        try:
            score=float(response.choices[0].message.content.strip())
        except ValueError:
            score=0.0
        reranked.append({
            "id":chunk["id"],
            "text":chunk["text"],
            "embedding_score":chunk["score"],
            "rerank_score":score

        })
    reranked.sort(key=lambda x:x["rerank_score"],reverse=True)
    return reranked[:top_k]


#MMR

def cosine_similarity(vec1:list[float],vec2:list[float])->list[float]:
    """calculate cosine similarity between two vectors.
    """
    dot=sum(a*b for a,b in zip(vec1,vec2))
    mag1=sum(a**2 for a in vec1)**0.5
    mag2=sum(b**2 for b in vec2)**0.5
    if mag1==0 or mag2==0:
        return 0
    return dot/(mag1*mag2)

def mmr_retrieve(query:str, fetch_k:int=None, top_k:int=None,diversity:float=0.5)->list[dict]:
    """
    Maximum Marginal Relevance retrieval.
    Balances relevance AND diversity.
    """
    top_k=top_k or settings.TOP_K
    fetch_k=fetch_k or settings.FETCH_K
    query_embedding=get_embedding(query)
    #Fetch more candidates than needed
    results=execute_query("""
        WITH scored AS(
            SELECT id,
            content,
            embedding<=>%s::vector AS distance
            FROM documents
            ORDER BY distance 
            LIMIT %s)
        SELECT id,
            content,
            1- distance AS similarity
        FROM scored 
        ORDER BY similarity DESC
           """,(query_embedding,fetch_k))
    if not results:
        return []
    #Build candidates with embeddings for MMR
    candidates=[]
    for row in results:
        embedding=get_embedding(row["content"])
        candidates.append({
            "id":row["id"],
            "text":row["content"],
            "score":row["similarity"],
            "embedding":embedding

        })
    #MMR Selection
    selected=[]
    remaining=candidates.copy()
    while len(selected)<top_k and remaining:
        if not selected:
            best=remaining[0]
        else:
            best=None
            best_mmr_score=-999
            for candidate in remaining:
                relevance=candidate["score"]
                max_sim=max(
                    cosine_similarity(candidate["embedding"], s["embedding"]) for s in selected
                )
                mmr_score=(
                    (1-diversity)*relevance-diversity*max_sim
                )
                if mmr_score>best_mmr_score:
                    best_mmr_score=mmr_score
                    best=candidate
        selected.append(best)
        remaining.remove(best)
    return [
        {
            "id":c["id"],
            "text":c["text"],
            "score":c["score"]
        }
        for c in selected
    ]

#MAIN RETRIEVE FUNCTION
#Combines MMR + RERANKING

def retrieve(query)->list[dict]:
    """
    Production retrieval pipeline:
    1.MMR - fetch diverse candidates
    2.Rerank - score by actual answer quality.
    3.Return top_k relevant diverse chunks
    """
    #Step 1- MMR retrieval
    mmr_chunks=mmr_retrieve(
        query=query,
        top_k=settings.FETCH_K,
        fetch_k=settings.FETCH_K*2,
        diversity=0.5
    )
    if not mmr_chunks:
        return []
    #Step 2 - Rerank MMR results
    reranked=rerank(
        query=query,
        chunks=mmr_chunks,
        top_k=settings.TOP_K
    )
    return reranked
from config import settings
from openai import OpenAI
import psycopg2


client=OpenAI(api_key=settings.OPENAI_API_KEY)

# -----------------DATABASE CONNNECTION--------------------------------
def get_db():
    return psycopg2.connect(
        host=settings.DB_HOST,
        port=settings.DB_PORT,
        database=settings.DATABASE,
        user=settings.USER,
        password=settings.PASSWORD

    )

# ----------------EMBEDDING-------------------------------------------------
def get_embedding(text:str)->list[float]:
    response=client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

# -----------------TECHNIQUE1-------------------------------------------------
def basic_retrieve(query:str,top_k:int=5)->list[dict]:
    """Original retrieval - pure embedding similarity."""
    query_embedding=get_embedding(query)
    conn=get_db()
    cursor=conn.cursor()
    cursor.execute("""
            WITH scored AS(
            SELECT content,
             
                embedding<=>%s::vector AS distance 
            FROM documents
            ORDER BY distance
            LIMIT %s)
            SELECT content, 1-distance AS similarity
            FROM scored
            ORDER BY similarity DESC
            """,(query_embedding,top_k))
    results=cursor.fetchall()
    cursor.close()
    conn.close()
    return [{"text":row[0], "score":row[1]} for row in results]

# -----------------TECHNIQUE2- RERANKING-------------------------------------

def rerank(query:str,chunks:list[dict],top_k:int=3)->list[dict]:
    """
    Use LLM to rerank retrieved chunks by relevance to query.
    More accurate than embedding similarity alone.
    
    We ask the LLM to score each chunk 0-10
    "How well does this answer the query?"
    """
    reranked=[]
    for chunk in chunks:
        response=client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role":"system",
                 "content":"""you are a relevance scorer.
                 Score how well the given text answers the query.
                 Respond with ONLY a number between 0 to 10.
                 0=completely irrelevant
                 10=perfectly answers the query
                 No explanation. just the number."""},
                 {
                     "role":"user",
                     "content":f"Query:{query}\n\n Text:{chunk["text"]}"
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
            "text":chunk['text'],
            "embedding_score":chunk["score"],
            "rerank_score":score
        })
    # sort by rerank score
    reranked.sort(key=lambda x:x["rerank_score"],reverse=True)
    return reranked[:top_k]
# ------------------TECHNIQUE3-MMR--------------------------------------
def cosine_similarity(vec1:list[float],vec2:list[float])->float:
    dot=sum(a*b for a, b in zip(vec1, vec2))**0.5
    mag1=sum(a**2 for a in vec1)**0.5
    mag2=sum(b**2 for b in vec2)**0.5
    if mag1==0 or mag2==0:
        return 0
    return dot/(mag1*mag2)


def mmr_retrieve(query:str,top_k:int=3,fetch_k:int=10,diversity:float=0.5)->list[dict]:
    """
    Maximum Marginal Relevance retrieval.
    Balances relevacne AND diversity.
    
    diversity=0.0-> pure relevance(same as basic retrieval).
    diversity=1.0 -> pure diversity(ignores relevance).
    diversity=0.5 -> balanced(recommended)
    """
    query_embedding=get_embedding(query)
    # step1: Fetch more candidate than we need
    conn=get_db()
    cursor=conn.cursor()
    cursor.execute(
        """
        WITH scored AS (
            SELECT content, embedding<=>%s::vector AS distance
            FROM documents
            ORDER BY distance
            LIMIT %s)
        SELECT content, 1-distance AS similarity
        FROM scored 
        ORDER BY similarity DESC
        """,(query_embedding,fetch_k)
    )
    results=cursor.fetchall()
    cursor.close()
    conn.close()

    candidates=[]
    for row in results:
        embedding=get_embedding(row[0])
        candidates.append({
            "text":row[0],
            "score":row[1],
            "embedding":embedding
        })

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
                max_sim_to_selected=max(
                    cosine_similarity(
                        candidate["embedding"],
                        s["embedding"]
                    ) 
                    for s in selected
                    )
            #MMR formula
            # High relevance + Low similarity to selected=good
                mmr_score=(
                    (1-diversity)*relevance
                    -diversity * max_sim_to_selected
                ) 
                if mmr_score>best_mmr_score:
                    best_mmr_score=mmr_score
                    best=candidate
        selected.append(best)
        remaining.remove(best)
    return [{"text":c["text"], "score":c["score"]} for c in selected]


#--------------------GENERATION-------------------------------------------------------------

def generate_answer(query:str,chunks:list[dict])->str:
    context="\n\n".join([c["text"] for c in chunks])
    system_prompt="""you are a helpful company assistant.
    Answer using ONLY the context below.
    If answer not in context, say "i don't have information about that."
    Context:
    {context}""".format(context=context)
    response=client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role":"system", "content":system_prompt},
            {"role":"user", "content":query},
        ],
        temperature=0.1,
        max_tokens=300,
    )
    return response.choices[0].message.content


#------------------------COMPARE ALL THREE---------------------------------------------------
def compare_retrival(query:str):
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"{'='*60}")
    #Basic retrieval
    print("\n-----BASIC RETRIEVAL---------")
    basic_chunks=basic_retrieve(query, top_k=3)
    for i,c in enumerate(basic_chunks):
        print(f" Chunk {i+1} (score :{c["score"]:.4f}):{c["text"][:80]}.....")
    basic_answer=generate_answer(query,basic_chunks)
    print(f"Answer :{basic_answer}")

    #Reranking
    print("\n------WITH RERANKING---------")
    candidates=basic_retrieve(query, top_k=5)
    reranked_chunks=rerank(query,candidates,top_k=3)
    for i,c in enumerate(reranked_chunks):
        print(f"Chunk {i+1} "
              f"(embed: {c["embedding_score"]:.4f}, "
              f"rerank: {c["rerank_score"]:.1f}): "
              f"{c["text"][:80]}.....")
    rerank_answer=generate_answer(query, reranked_chunks)
    print(f"Answer : {rerank_answer}")

    ##MMR
    print("\n-----WITH MMR-----")
    mmr_chunks=mmr_retrieve(query, top_k=3, fetch_k=5, diversity=0.5)
    for i,c in enumerate(mmr_chunks):
        print(f"Chunk {i+1} (score: {c["score"]:.4f}):{c["text"][:80]}......")

    mmr_answer=generate_answer(query, mmr_chunks)
    print(f"Answer: {mmr_answer}")

#------------RUN IT-------------------------------------------------------
compare_retrival("How long do refunds take?")
compare_retrival("what are the company policies?")

    

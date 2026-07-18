from config import settings
from openai import OpenAI
import psycopg2
from psycopg2.extras import execute_values




client=OpenAI(api_key=settings.OPENAI_API_KEY)
#  ------------------DATABASE CONNECTION----------------------

def get_db_connection():
    ''' create connection to pgvector database'''
    return psycopg2.connect(
        host=settings.DB_HOST,
        port=settings.DB_PORT,
        database=settings.DATABASE,
        user=settings.USER,
        password=settings.PASSWORD
    )

#--------------EMBEDDING----------------------------------------
def get_embedding(text:str):
    embeddings=client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return embeddings.data[0].embedding

# -----------INDEXING-store documents in PostgreSQL------------
def index_documents(documents:list[str]):
    conn=get_db_connection()
    cursor=conn.cursor()
    print("indexing documents into postgreSQL")

    for i, doc in enumerate(documents):
        embedding=get_embedding(doc)
        cursor.execute(
            """insert into documents(content,embedding)
             VALUES (%s,%s)
            on conflict (content) do nothing""",
            (doc,embedding)

        )
        print(f"INDEXED document {i+1}/{len(documents)}")
    conn.commit()
    cursor.close()
    conn.close()
    print("All documents indexed.\n")

# -------------RETRIVAL-search using pgvector-------------------
def retrieve(query:str, top_k:int=2):
    '''find most relevant documents using pgvector cosine search 
    this replaces our manual cosine similarity loop'''
    query_embedding=get_embedding(query)
    conn=get_db_connection()
    cursor=conn.cursor()
    cursor.execute("""
            with scored as (
                   select content,
                   embedding<=>%s::vector as distance
                from documents
                order by distance
                limit %s
                   )
            select content,
                   1-distance as similarity
                from scored
                order by similarity DESC
                   """,
                   (query_embedding,top_k)
                   
    )
    results=cursor.fetchall()
    cursor.close()
    conn.close()
    return [
        {"text":row[0],"score":row[1]} for row in results
    ]


# -------------GENERATION----------------------------------------------
def generate_answer(query:str,retrievfed_chunks:list[dict])->str:
    """generate answer using retrieved context."""
    context="\n\n".join([chunk["text"] for chunk in retrievfed_chunks])
    system_prompt="""you are a helpful company assistant.
    Answer questions using ONLY the context provided below.
    if the answer is not in the conotext, say "i don't have the information about that."
    Do not use any outside knoledge.
    Context:
    {context}""".format(context=context)
    response=client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role":"system","content":system_prompt},
            {"role":"user", "content":query}
        ],
        max_tokens=300,
    )
    return response.choices[0].message.content



# ----------FULL RAG PIPELINE------------------------------------
 

def rag_query(query:str):
    """complete  RAG pipeline using PostgreSQL"""
    print(f"query:{query}")
    print("-"*50)
    relevant_chunks=retrieve(query,top_k=2)
    for i,chunk in enumerate(relevant_chunks):
        print(f"chunk {i+1} (score: {chunk['score']:.4f}):")
        print(f"{chunk['text'][:100]}....")
    answer=generate_answer(query,relevant_chunks)
    print(f"\n Answer : {answer}")
    print("="*50 + "\n")



# -----------RUN IT------------------------------------------------


documents = [
    """Our refund policy allows customers to return products within 30 days 
    of purchase for a full refund. Products must be in original condition. 
    Refunds are processed within 5-7 business days to the original payment method.""",

    """Employee leave policy: Full-time employees are entitled to 20 days 
    of paid annual leave per year. Leave must be approved by the direct manager 
    at least 2 weeks in advance. Unused leave can be carried over to next year 
    up to a maximum of 10 days.""",

    """Our IT security policy requires all employees to use strong passwords 
    of at least 12 characters. Passwords must be changed every 90 days. 
    Two-factor authentication is mandatory for all company systems.""",

    """Travel expense policy: Employees traveling for business can claim 
    accommodation up to 5000 rupees per night. Meals are reimbursed up to 
    1000 rupees per day. All expenses must be submitted within 30 days 
    with original receipts.""",

    """Python backend engineers at our company use FastAPI for REST APIs, 
    PostgreSQL for primary database, and Redis for caching. All services 
    are containerized using Docker and deployed on AWS.""",
]

index_documents(documents)
rag_query("What is the refund policy?")
rag_query("What technology stack does the company use?")
rag_query("What is the capital of France?")

   


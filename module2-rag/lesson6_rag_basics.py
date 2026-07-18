import os
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

client=OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

document=[
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


def get_embedding(doc:str):
    """Get embedding vector for a piece of text."""
    response=client.embeddings.create(
        model="text-embedding-3-small",
        input=doc,


    )
    # print(response)
    return response.data[0].embedding



def build_knowlege_base(docs:list[str])->list[dict]:
    """
    Embed all documents and store with their text.
    Returns list of {text, embedding} dicts.
    In production: store these in PostgreSQL with pgvector.
    """
    knowledge_base=[]
    for i, doc in enumerate(docs):
        embedding=get_embedding(doc)
        knowledge_base.append({
               "index":i,
               "text":doc,
               "embedding":embedding
          })
        print(f"emebedded document {i+1}/{len(docs)}")
    print(f"knowledge base  ready: {len(knowledge_base)} docunement indexed.\n")
    return knowledge_base

# knowledge_base=build_knowlege_base(document)
# for embeding in knowledge_base:
#     print(embeding)

def cosine_similarity(vec1:list[float], vec2:list[float])->float:
    """calculate cosine similarity between vectors"""
    
    dot_product=sum(a*b for a,b in zip(vec1,vec2))
    mag1=sum(a**2 for a in vec1)**0.5
    mag2=sum(b**2 for b in vec2)**0.5
    if mag1==0 or mag2==0:
        return 0
    return dot_product/(mag1*mag2)


# ------find top k most relevent chunks

def retrieve(query:str, knowledge_base:list[dict],top_k:int=2)->list[dict]:
    embeding=get_embedding(query)

    scored=[]
    for doc in knowledge_base:
        score=cosine_similarity(embeding, doc["embedding"])
        scored.append(
            {"text":doc["text"],
             "score":score}
        )
    scored.sort(key=lambda x:x["score"], reverse=True)
    return scored[:top_k]


def generate_answer(query:str,retrieved_chunks:list[dict])->str:
    '''generate answer using retrieved chunks'''

    context="\n\n".join([chunk['text'] for chunk in retrieved_chunks])
    system_prompt = """You are a helpful company assistant.
Answer questions using ONLY the context provided below.
If the answer is not in the context, say "I don't have information about that."
Do not use any outside knowledge.

Context:
{context}""".format(context=context)
    response=client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{
            "role":"system", "content":system_prompt},
           { "role":"user", "content":query
        }
        ],
        temperature=0.3,
        max_tokens=300
    )
    return response.choices[0].message.content

# ----------------RAG QUERY--------------

def rag_query(query:str,knowledge_base:list[dict]):
    '''complete rag pipeline    
      1.Retrieve relevent chunks
      2.Generate answer from chunks'''
    relevant_chunks=retrieve(query,knowledge_base, top_k=2)

    for i, chunk in enumerate(relevant_chunks):
        print(f" chunk {i+1} scores:  {chunk['score']}")
        print(f"\n {chunk['text'][:100]}")
    answer=generate_answer(query, relevant_chunks)
    print(f"\n Answer :{answer}")
    print("="*50 + "\n")



# =====RUN IT=============
kb=build_knowlege_base(document)

rag_query("what is the refund policy?",kb)
rag_query("What technology stack does the company use?", kb)
rag_query("What is the capital of France?", kb)




          
from openai import OpenAI
from config import settings

client=OpenAI(api_key=settings.OPENAI_API_KEY)



def generate_answer(query:str, chunks:list[dict])->dict:
    """Generate grounded answer from retrieved chunks.
    Returns dict with answer, source , and  metadata.
    """
    if not chunks:
        return {
            "answer":"I don't have information about that.",
            "sources":[],
            "chunks_used":0,
            "tokens_used":0
        }

    #Build context from chunks
    context_parts=[]
    for i, chunk in enumerate(chunks):
        context_parts.append(f"[source {i+1}]\n {chunk["text"]}")
    context="\n\n".join(context_parts)
    system_prompt="""You are a helpful company assistent.
    Answer questions using ONLY the context provided below.
    if the answer is not in the context, say exactly:
    "I don't have information about that"
        
        
    Rules:
    - Never use outside knowledge
    - Never make assumptions beyond the context
    - Be concise and direct
    - If context partially answers - answer what you  can and state what is missing
    Context:
    {context}""".format(context=context)
    response=client.chat.completions.create(
        model=settings.LLM_MODEL,
        messages=[
            {"role":"system","content":system_prompt},
            {"role":"user", "content":query}
        ],
        temperature=0.1,
        max_tokens=500,
    ) 
    answer=response.choices[0].message.content
    return {
        "answer":answer,
        "sources":[
            {"id":chunk.get("id"),
             "text":chunk["text"][:200],
             "score":round(chunk.get("rerank_score",chunk.get("score",0)),4)
             } for chunk in chunks
        ],
        "chunks_used":len(chunks),
        "tokens_used":response.usage.total_tokens
    }
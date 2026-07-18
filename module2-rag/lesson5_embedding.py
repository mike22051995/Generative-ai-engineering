import os
from dotenv import load_dotenv

from openai import OpenAI

load_dotenv()

client=OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def get_embedding(text:str)->list[float]:
    '''convert text to embedding vector'''

    response=client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding



def cosine_similarity(vec1:list[float],vec2:list[float])->float:
    """calculate cosine similarity between two vectors.
    Returns a number between -1 to 1.
    1=identical direction (Same meaning)
    0=perpendicular (no relationship)
    -1=opposite direction (opposite meaning)"""
    dot_product=sum(a*b for a, b in zip(vec1,vec2))
    mag1=sum(a**2 for a in vec1)**0.5
    mag2=sum(b**2 for b in vec2)**0.5
    if mag1==0 or mag2==0:
        return 0
    return dot_product/(mag1*mag2)

texts=[
    "what is our refund policy?",
    "how do i return a product?",
    "tell me about your return process?",
    "how do i cook pasta?",
    "what are the ingredients for pizza?"
]


print("\n----------GETTING EMBEDDING---------------")

embeddings={}
for text in texts:
    embeddings[text]=get_embedding(text)
    print(f"✓ {text}")
    
print("\n ---------------------similarity matrix------------------\n")

reference=texts[0]
ref_embedding=embeddings[reference]
print(f"Reference: '{reference}'\n")



for text in texts[1:]:
    sim=cosine_similarity(ref_embedding,embeddings[text])
    print(f"       vs '{text}")
    print(f"     similarity: {sim:.4f}\n")




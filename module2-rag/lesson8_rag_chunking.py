import re


#------------FIX SIZE CHUNKING-----------------------

def fixed_size_chunk(text:str,chunk_size:int=200)->list[str]:
    """split every chunk_size characters, ignore sentence boundaries"""
    chunks=[]
    for i in range(0,len(text),chunk_size):
        chunks.append(text[i:i+chunk_size])
    

    return chunks

# -----------SENTENCE BASED----------------------------

def sentence_chunk(text:str,sentence_per_chunk:int=2)->list[str]:
    """split on sentence boundaries. Respects complete thoughts."""
    sentences=re.split(r'(?<=[.!?])\s+',text)
    chunks=[]
    for i in range(0,len(sentences),sentence_per_chunk):
        chunk=" ".join(sentences[i:i+sentence_per_chunk])
        chunks.append(chunk)
    return chunks


# .----------RECURSIVE WITH OVERLAP-----------------------

def recursive_chunk(text:str,max_size:int=200,overlap:int=50)->list[str]:
    """Split on natural boundaries in order:
    paragraph ->sentence ->word
    Add overlap to preserve boundary context."""
    chunks=[]
    current=" "
    paragraphs=text.split("\n\n")
    for para in paragraphs:
        if len(current)+len(para)<max_size:
            current+=para + " "
        else:
            if current:
                chunks.append(current.strip())
            if len(para)>max_size:
                sentences=re.split(r'(?<=[.!])\s+',para)
                for sent in sentences:
                    if len(current)+len(sent)<max_size:
                        current+=sent + " "
                    else:
                        if current:
                            chunks.append(current.strip())
                        current=sent+ " "
            else:
                current=para + " "
    if current:
        chunks.append(current.strip())
    chunks = [c for c in chunks if c.strip()]

    # Add overlap between chunks
    if overlap>0 and len(chunks)>1:
        overlapped=[chunks[0]]
        for i in range(1,len(chunks)):
            prev_end=chunks[i-1][-overlap:]
            overlapped.append(prev_end+ " "+chunks[i])
        return overlapped
   
    return chunks


# ---------------SHOW CHUNKS HELPER------------------------------------
def show_chunks(name:str, chunks:list[str]):
    print(f"\n{'='*50}")
    print(f"Strategy :{name}")
    print(f"Total Chunks :{len(chunks)}")
    print(f" {'='*50}")
    for i, chunk in enumerate(chunks):
        print(f"\n chunk {i+1} ({len(chunk)} chars): ")
        print(chunk)
        print("-"*30)

# ---------------TEST TEXT-------------------------------------------------


sample_text="""
Our refund policy allows customers to return products within 30 days of purchase for a full refund. Products must be original condition.
Refunds are processed within 5-7 business days to the original payment method.

Employee leave policy: Full-time employee are entitled to 20 days of paid annual leave per year. Leave must be approved by the direct 
manager at least 2 weeks in advance. Unused leave can be carried over to next year up to a maximum of 10 days.

Our IT security policy requires all employees to use strong passwords of at least 12 characters. Passwords must be changed every 90 days.
Two-factor authentication is mandatory for company systems."""

# ------------RUN ALL THREE------------------------------------------------
show_chunks("Fixed Size(200)",fixed_size_chunk(sample_text,chunk_size=200))

show_chunks(
    "Sentence Based (2 sentences)",sentence_chunk(sample_text,sentence_per_chunk=2)
)

show_chunks("Recursive with overlap (max=200, overlap=50)",recursive_chunk(sample_text, max_size=200,overlap=50))

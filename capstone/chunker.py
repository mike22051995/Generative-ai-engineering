import re 
from config import settings

def recursive_chunk(text:str, chunk_size:int=None,overlap:int=None)->list[str]:
    '''
    Split text using recursive strategy.
    Tries paragraph->sentence->word boundaries in order
    Add overlap between chunks to preserve the context.
    Use setting default if not specified.
    '''
    chunk_size=chunk_size or settings.CHUNK_SIZE
    overlap=overlap or settings.CHUNK_OVERLAP
    chunks=[]
    current=""
    #split on paragraph first
    paragraphs=text.split("\n\n")
    for para in paragraphs:
        para=para.strip()
        if not para:
            continue
        if len(current)+len(para)<chunk_size:
            current+=para+" "
        else:
            if current:
                chunks.append(current.strip())
            if len(para)>chunk_size:
                sentences=re.split(r'(?<=[.?!]\s++)',para)
                for sent in sentences:
                    if len(current)+len(sent)<chunk_size:
                        current+=sent+ " "
                    else:
                        if current:
                            chunks.append(current.strip())
                            curr=sent+" "
            else:
                current=para + " "
    if current.strip():
        chunks.append(current.strip())
    chunks=[c for c in chunks if c.strip()]
    #Add overlap between chunks

    if overlap>0 and len(chunks)>1:
        overlapped=[chunks[0]]
        for i in range(1,len(chunks)):
            prev_end=chunks[i-1][-overlap:]
            overlapped.append(prev_end+ " " +chunks[i])
        return overlapped
    return chunks

def chunk_document(text:str,metadata:dict=None)->  list[dict]:
    """
    Chunk a document and attach metadata to each chunk.
    Retun list of dict ready for embedding and storage.
    metadat=additional info about the document.
            e.g filename,upload date, document type
            """
    chunks=recursive_chunk(text)
    result=[]
    for i, chunk in enumerate(chunks):
        result.append({
            "content":chunk,
            "chunk_index":i,
            "total_chunks":len(chunks),
            "metadata":metadata or {}
        }
        )
    return result


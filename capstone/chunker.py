import re 
from config import settings

def recursive_chunk(text: str,
                    chunk_size: int = None,
                    overlap: int = None) -> list[str]:
    """
    Split text into chunks of roughly chunk_size characters.
    Uses sentence boundaries where possible.
    Adds overlap between chunks.
    """
    chunk_size = chunk_size or settings.CHUNK_SIZE
    overlap = overlap or settings.CHUNK_OVERLAP

    # Split into sentences first
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(current) + len(sentence) < chunk_size:
            current += sentence + " "
        else:
            if current.strip():
                chunks.append(current.strip())
            current = sentence + " "

    if current.strip():
        chunks.append(current.strip())

    # Remove empty chunks
    chunks = [c for c in chunks if len(c.strip()) > 20]

    # Add overlap
    if overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_end = chunks[i-1][-overlap:]
            overlapped.append(prev_end + " " + chunks[i])
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


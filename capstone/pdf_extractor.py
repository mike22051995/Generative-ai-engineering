import pymupdf as fitz

def extract_text_from_pdf(pdf_bytes:bytes)->str:
    """
    Extract all text from a PDF given its raw bytes.
    pdf_bytes = the raw binary content of the PDF file
    Returns: extracted text as a single string
    """

    doc=fitz.open(stream=pdf_bytes, filetype="pdf")
    text_parts=[]

    for page_num in range(len(doc)):
        page=doc[page_num]
        text=page.get_text()

        if text.strip():
            text_parts.append(text.strip())
    doc.close()
    full_text="\n\n".join(text_parts)
    return full_text

def get_pdf_info(pdf_bytes:bytes)->dict:
    """
    Get basic metadata about a PDF.
    Useful for logging and debugging.
    """
    doc=fitz.open(stream=pdf_bytes, filetype="pdf")
    info={
        "page_count":len(doc),
        "title":doc.metadata.get("title","unknown"),
        "author":doc.metadata.get("author","unknown"),
    }
    doc.close()
    return info


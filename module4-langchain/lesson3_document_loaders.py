import os
from dotenv import load_dotenv
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader,CSVLoader,WebBaseLoader
from langchain_core.documents import Document 

load_dotenv()

print("-"*50+"PDF LOADER"+"-"*50)

pdf_loader=PyPDFLoader(r"C:\Users\MUKESH PRASAD\OneDrive\Desktop\JOB SEARCH\Resume[01-06-2026].pdf")
pdf_docs=pdf_loader.load()

print(f"Total pages loaded: {len(pdf_docs)}")
print(f"Type of first element:{type(pdf_docs[0])}")
print(f"\nPage 1 content (first 200 chars):")
print(pdf_docs[0].page_content[:200])
print(f"\nPage 1 metadat: {pdf_docs[0].metadata}")

#LOADER 2 - CSV

print("-"*50+"CSV LOADER"+"-"*50)

csv_loader=CSVLoader("employee.csv")
csv_docs=csv_loader.load()

print(f"Total rows loaded: {len(csv_docs)}")
print(f"\nFirst row content:")
print(csv_docs[0].page_content)
print(f"\nFirst row metadata: {csv_docs[0].metadata}")

print(f"\nAll employees:")
for doc in csv_docs:
    print(doc.page_content)
    print("-"*10)

#LOADER 3 - Web load
print("-"*50 + "WEB LOADER"+"-"*50 + "\n")

web_loader=WebBaseLoader("https://en.wikipedia.org/wiki/FastAPI")
web_docs=web_loader.load()

print(f"Total documents laoded: {len(web_docs)}")
print(f"\nContent (first 300 chars):")
print(web_docs[0].page_content[:300])
print(f"\nMetadata: {web_docs[0].metadata}")

#KEY INSIGHT - same interface, different sources
print("\n----- ALL LAODERS RETURN THE SAME TYPE-----\n")
print(f"PDF laoder returns : {type(pdf_docs[0])}")
print(f"CSV laoder returns: {type(csv_docs[0])}")
print(f"Web laoder returns: {web_docs[0]}")
print("\nAll return Document objects - same pipeline works for all sources")
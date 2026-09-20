import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

#STEP 1: Documents ───
docs = [
    Document(page_content="Our refund policy allows customers to return products within 30 days of purchase for a full refund. Products must be in original condition. Refunds are processed within 5-7 business days.", metadata={"source": "refund_policy"}),
    Document(page_content="Employee leave policy: Full-time employees are entitled to 20 days of paid annual leave per year. Leave must be approved by manager 2 weeks in advance.", metadata={"source": "leave_policy"}),
    Document(page_content="IT security policy requires strong passwords of at least 12 characters. Passwords must be changed every 90 days. Two-factor authentication is mandatory.", metadata={"source": "security_policy"}),
    Document(page_content="Travel expense policy: accommodation up to 5000 rupees per night. Meals reimbursed up to 1000 rupees per day. Submit expenses within 30 days with receipts.", metadata={"source": "travel_policy"}),
    Document(page_content="Python backend engineers use FastAPI for REST APIs, PostgreSQL for database, and Redis for caching. All services containerized using Docker.", metadata={"source": "tech_stack"}),
]

#STEP 2: Chunk 
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)
print(f"Total chunks: {len(chunks)}")

#STEP 3: Embed and store
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.from_documents(chunks, embeddings)
print("Vector store created")

#STEP 4: Retriever 
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

#STEP 5: Prompt 
prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer using ONLY the context below.
If the answer is not in the context, say "I don't have information about that."

Context: {context}

Question: {question}
""")

#STEP 6: LLM 
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)

#STEP 7: Chain 
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

#STEP 8: Ask 
def ask(question: str):
    print(f"\nQuestion: {question}")
    answer = chain.invoke(question)
    print(f"Answer: {answer}")
    print("-" * 50)

ask("What is the refund policy?")
ask("How many leave days do employees get?")
ask("What is the capital of France?")
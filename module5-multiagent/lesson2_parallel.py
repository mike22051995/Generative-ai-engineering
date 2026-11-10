import os
from typing import Annotated,TypedDict
from operator import add
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, START,END

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
tavily=TavilySearch(max_results=2)

#STATE - with Annotated for parallel merging

class ParallelState(TypedDict):
    topic:str
    search_results:Annotated[list,add]
    final_answer:str

#PARALLEL NODE 1 - Web search

def web_search_node(state:ParallelState)->dict:
    print("\n[WEB SEARCH] Searching web...")

    topic=state['topic']
    results=tavily.invoke(topic)

    findings=[]
    items = results.get("results", []) if isinstance(results, dict) else results

    for r in items:
        if isinstance(r, dict):
            findings.append(f"[WEB] {r.get('content', '')[:300]}")
        else:
            findings.append(f"[WEB] {str(r)[:300]}")
    print(f"[WEB SEARCH] Found {len(findings)} results")
    return {"search_results":findings}

#PARALLEL NODE 2 - Company Policy Search
def policy_search_node(state:ParallelState)->dict:
    print("\n[POLICY SEARCH] Searching company policies...")

    policies={
        "rag":"Our RAG implementation uses pgvector with HNSW indexing."
        "Chunk size is 500 tokens with 50 token overlap. We use MMR retrieval with parallel reranking.",
        "llm":"We use GPT-3.5-turbo for generation and text-embedding-3-small for embeddings. Temperatur is 0.1 for factual tasks.",
        "caching":"Redis caching with 1hour TTL. Cache invalidated on new document upload. Average cache hit rate is 78%.",
        "deployment":"All services containerized with Docker Compose. PostgreSQL on port 5433, Redis on 6379, FastAPI on 8000.",

    }
    topic_lower=state['topic'].lower()
    findings=[]
    for key,value in policies.items():
        if key in topic_lower or any(word in topic_lower for word in key.split()):
            findings.append(f"[POLICY] {value}")
    if not findings:
        findings.append("[POLICY] no specific company policy found for this topic.")
    print(f"[POLICY SEARCH] Found {len(findings)} policies")
    return {"search_results":findings}

#PARALLEL NODE 3 - Technical knowledge Base

def knowledge_base_node(state:ParallelState)->dict:
    print("\n [KNOWLEDGE BASE] Searching internal knowledge.....")
    response=llm.invoke([
        {
            "role":"system",
            "content":""""You are a technical knowledge base for RAG and AI systems.
            Provide 2-3 specific technical facts about the given topic.
            Be concise and factual. Each fact on a new line starting with [KB]."""

        },
        {
            "role":"user",
            "content":f"Provide technical facts about:{state['topic']}"
        }
    ])

    facts=[line for line in response.content.split('\n') if line.strip()]
    print(f"[KNOWLEDGE BASE] Generated {len(facts)} facts")
    return {"search_results":facts}

#AGGREGATOR NODE - runs after all parallel node complete

def aggregator_node(state:ParallelState)->dict:
    print("\n[AGGREGATOR] combining all results...")

    all_results=state['search_results']
    print(f"[AGGREGATOR] Total results to synthesize: {len(all_results)}")

    combined="\n".join(all_results)
    response=llm.invoke([
        {
            "role":"system",
            "content":"""You are a technical synthesizer.
            Synthesize ONLY from the sources provided below. Do not add outside knowledge.
            Tag each point with its source type: [WEB], [POLICY], or [KB].
            Put company policy facts first and state them exactly.
            If sources conflict, say so explicitly."""
        },
        {
            "role":"user",
            "content":f"Topic:{state['topic']}\n\nResults from all sources:\n{combined}"
        }
    ])
    print("[AGGREGATOR] Synthesis complete")
    for r in state["search_results"]:
        print(r[:80])
    return {"final_answer":response.content}

graph=StateGraph(ParallelState)

graph.add_node("web_search", web_search_node)
graph.add_node("policy_search",policy_search_node)
graph.add_node("knowledge_base",knowledge_base_node)
graph.add_node("aggregator", aggregator_node)

#fan-out - START connects to all three parallel nodes
graph.add_edge(START,"web_search")
graph.add_edge(START,"policy_search")
graph.add_edge(START,"knowledge_base")

#Fan-in - all three parallel nodes connect to aggregator
graph.add_edge("web_search","aggregator")
graph.add_edge("policy_search","aggregator")
graph.add_edge("knowledge_base","aggregator")

graph.add_edge("aggregator",END)

pipeline=graph.compile()


def parallel_search(topic:str):
    print(f"\n{"="*60}")
    print(f"Topic:{topic}")
    print(f"\n{"="*60}")
    result=pipeline.invoke({
        "topic":topic,
        "search_results":[],
        "final_answer": ""
    })
    print(f"\n{"="*60}")
    print("FINAL SYNTHESIZED ANSWER:")
    print(f"\n{"="*60}")
    print(result['final_answer'])
    print(f"\nTotal sources used: {len(result['search_results'])}")

parallel_search("RAG caching stragegies and best practices")
    
        

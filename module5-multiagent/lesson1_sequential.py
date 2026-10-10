import os
from typing import TypedDict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph,END

load_dotenv()

llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)
search=TavilySearch(max_result=3)

#STATE
class ContentState(TypedDict):
    topic:str
    research:str
    draft:str
    review:str
    final:str

#NODE 1 - Research Agent

def research_node(state:ContentState)->dict:
    print("\n [RESEARCHER] Starting research....")
    topic=state["topic"]
    results=search.invoke(topic)
    research_text=f"Research findings for: {topic}\n\n"
    if isinstance(results,list):
        for r in results:
            if isinstance(r,dict):
                research_text+=f"Source:{r.get('url','unknown')}/n"
                research_text+=f"Content:{r.get('content',r.get('raw_content',""))}\n\n"
            else:
                research_text+=f"{r}\n\n"
    else:
        research_text+=str(results)
    print(f"[RESEARCHER] Research complete")
    return {"research":research_text}

#NODE 2 - Writer Agent

def writer_node(state:ContentState)->dict:
    print("\n[WRITER] writing draft...")
    response=llm.invoke([
        {
            "role":"system",
            "content":"""You are an expert technical writer.
            Write a clear, structured article based on the research provided.
            Format with headers and bullet points.
            Keep it concise - maximum 400 words."""

        },
        {
            "role":"user",
            "content":f"Topic: {state['topic']}\n\nResearch:\n{state['research']}"
        }

    ])
    print("[WRITER] Draft complete")
    return {"draft":response.content}

#NODE 3 - Review Report
def review_node(state:ContentState)->dict:
    print("\n[REVIEWER] Reviewing draft.....")
    response=llm.invoke([
        {
            "role":"system",
            "content":"""you are a senior technical editor.
            Review the draft for:
            -Accuracy
            -Clarity
            -Missing important points
            -Overall quality
            Give a score out of 10 and specific feedback."""

        },
        {
            "role":"user",
            "content":f"Topic: {state['topic']}\n\nDraft to review:\n{state['draft']}"
        }
    ])
    print("[REVIEWER] Review complete")
    return {"review":response.content}

#NODE 4 - Final Assembly Agent
def final_node(state:ContentState)->dict:
    print("\n[FINAL] Assemnling final output...")
    response=llm.invoke([
        {
            "role":"system",
            "content":"""you are a content director.
            Given a draft and its review feedback:
            1. Apply the reviewer's suggestions
            2. Polish the final version
            3. Add a brief executive summary at the top"""

        },
        {
            "role":"user",
            "content":f"""
            Topic: {state['topic']}
            Original Draft: {state['draft']}
            Review Feedback: {state['review']}
            please create the final polished vsersion."""
        }
    ])
    print("[FINAL] Final content ready")
    return {"final":response.content}

#BUILD THE GRAPH
graph=StateGraph(ContentState)
graph.add_node("researcher", research_node)
graph.add_node("writer",writer_node)
graph.add_node("reviewer", review_node)
graph.add_node("final", final_node)

graph.add_edge("researcher", "writer")
graph.add_edge("writer", "reviewer")
graph.add_edge("reviewer","final")
graph.add_edge("final",END)

graph.set_entry_point("researcher")
pipeline=graph.compile()

#RUN THE PIPELINE

def create_content(topic:str):
    print(f"\n{"="*60}")
    print(f"Creating content for: {topic}")
    initial_state={
        "topic":topic,
        "research":"",
        "draft":"",
        "review":"",
        "final":""
    }
    result=pipeline.invoke(initial_state)
    print(f"\n{"="*60}")
    print("FINAL CONTENT:")
    print('='*60)
    print(result["final"])
    print(f"\n{"="*60}")
    print("REVIEW FEEDBACK")
    print("="*60) 
    print(result["review"])

    return result

create_content("Best practices for RAG systems in production")


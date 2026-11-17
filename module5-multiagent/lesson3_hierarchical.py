import os
from operator import add
from typing import Annotated,TypedDict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph,START,END


load_dotenv()

llm=ChatOpenAI(model="gpt-4o-mini", temperature=0)
tavily=TavilySearch(max_results=5)

MAX_STEPS=8
MAX_REVISIONS=5
ALLOWED={"researcher","writer","reviewer", "finish"}

class TeamState(TypedDict):
    request:str
    research:str
    draft:str
    review:str
    next_worker:str
    steps:int
    revisions:int
    path:Annotated[list,add]
    needs_research:bool
    needs_review:bool

def plan_request(request: str) -> dict:
    r = request.lower()
    return {
        "needs_research": any(w in r for w in ("research", "current", "latest")),
        "needs_review": any(w in r for w in ("review", "fact check", "fact-check", "verify")),
    }


def decide(state: TeamState) -> str:
    if not state["research"] and state["needs_research"]:
        return "researcher"
    if not state["draft"]:
        return "writer"
    if state["needs_review"] and not state["review"]:
        return "reviewer"
    if state["review"].upper().startswith("NEEDS REVISION") and state["revisions"] < MAX_REVISIONS:
        return "writer"
    return "finish"


def orchestrator_node(state: TeamState) -> dict:
    steps = state["steps"]
    if steps >= MAX_STEPS:
        choice = "finish"
        print("[ORCHESTRATOR] step limit reached -> finish")
    else:
        choice = decide(state)
    print(f"[ORCHESTRATOR] step {steps + 1}: next = {choice}")
    return {"next_worker": choice, "steps": steps + 1, "path": [f"orchestrator:{choice}"]}

def researcher_node(state:TeamState)->dict:
    print("[RESEARCHER] searching.....")
    results=tavily.invoke(state['request'])
    items=results.get('results',[]) if isinstance(results,dict) else results
    text="\n".join(r.get('content',"")[:300] for r in items if isinstance(r,dict))
    return {"research":text,"path":["researcher"]}

def writer_node(state:TeamState)->dict:
    print("[WRITER] writing...")
    context=f"Request:{state['request']}"
    if state['research']:
        context+=f"\n\nResearch to use:\n{state['research']}"
    if state['review']:
        context+=f"\n\nReviewer feedback to address:\n{state['review']}"
    response=llm.invoke([
        {"role":"system", "content":"you are a concise technical writer. Write what the request asks for. Maximum 150 words."},
        {"role":"user","content":context},
    ])
    return {"draft":response.content,"review":"","revisions":state['revisions']+1,"path":["writer"]}

def reviewer_node(state:TeamState)->dict:
    print("[REVIEWER] reviewing...")
    response=llm.invoke([
        {"role":"system", "content":"you are a strict editor. Start your answer with APPROVED or NEEDS REVISION,then give one or two sentences of feedback."},
        {"role":"user","content":f"Request:{state['request']}\n\nDraft:\n{state['draft']}"},

    ])
    return {"review":response.content, "path":["reviewer"]}

def route(state:TeamState)->str:
    return state['next_worker']


graph=StateGraph(TeamState)
graph.add_node("orchestrator",orchestrator_node)
graph.add_node("researcher",researcher_node)
graph.add_node("writer",writer_node)
graph.add_node("reviewer", reviewer_node)

graph.add_edge(START, "orchestrator")
graph.add_conditional_edges(
    "orchestrator",
    route,
    {"researcher":"researcher","writer":"writer","reviewer":"reviewer","finish":END},
)
for worker in ("researcher","writer","reviewer"):
    graph.add_edge(worker,"orchestrator")

team=graph.compile()

def run(request: str):
    print(f"\n{'='*60}\nRequest: {request}\n{'='*60}")
    result = team.invoke(
        {"request": request, "research": "", "draft": "", "review": "",
         "next_worker": "", "steps": 0, "revisions": 0, "path": [],
         **plan_request(request)},
        config={"recursion_limit": 20},
    )
    print(f"\nPATH TAKEN: {' -> '.join(result['path'])}")
    print(f"\nFINAL DRAFT:\n{result['draft']}")


run("Write a 100 word note on why RAG systems need caching.")
run("Research current best practices for RAG chunking and write a short review summary.")

import os
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool

load_dotenv()

llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

#TOOLS - same as lesson 4
@tool
def calculate(expression)->str:
    """
    Useful for doing math calculations.
    Input should be a valid python math expression.
    Example: '2+2' or '100*1.18'
    """
    try:
        result=eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def get_current_date(input:str="")->str:
    """
    Returns the current date and time.
    Use when user asks about today's date or current time.
    """
    return datetime.now().strftime("%Y-%m-%d %H:M:%S")

@tool
def search_company_policy(query:str)->str:
    """
    Search company policies and documents.
    Use when user asks about refund, leavem travel, security or tech policies.
    Input should be the topic to search for.
    """
    policies={
        "refund": "Customers can return products within 30 days for a full refund. Refunds processed in 5-7 business days.",
        "leave": "Full-time employees get 20 days paid annual leave. Manager approval needed 2 weeks in advance.",
        "travel": "Accommodation up to 5000 rupees per night. Meals up to 1000 rupees per day.",
        "security": "Passwords must be 12+ characters. Changed every 90 days. Two-factor authentication mandatory.",
        "tech": "Backend uses FastAPI, PostgreSQL, Redis. All services containerized with Docker."
    }
    query_lower=query.lower()
    for key, value in policies.items():
        if key in query_lower:
            return value
    return "No specific policy found for this query."

#AGENT WITH MEMORY
tools=[calculate, get_current_date,search_company_policy]
agent=create_agent(
    model=llm,
    tools=tools,
    system_prompt="""You are a helpful company assistant with access to tools.
    Use tools when needed to answer questions accurately.
    Remeber previous parts of the conversation when answering follow-up questions."""
)

#CONVERSATION HISTORY - the memory
conversation_history=[]

def ask_agent(question:str)->str:
    """Ask agent a question - maintains converstaion history."""
    print(f"\n{"="*60}")
    print(f"Human:{question}")

    conversation_history.append({"role":"user", "content":question})

    result=agent.invoke({"messages":conversation_history})
    # print(f"Result type: {type(result)}")
    # print(f"Result keys: {result.keys() if hasattr(result, 'keys') else 'not a dict'}")
    # print(f"Result: {result}")
    final_answer=result["messages"][-1].content

    conversation_history.append({"role":"assistant", "content":final_answer})
    print(f"Agent:{final_answer}")
    return final_answer

#TEST MEMORY

print("=== Testing Agent Memory ===\n")

# Turn 1 — introduce yourself
ask_agent("My name is Mukesh and I work at Nutanix as a Senior Backend Engineer")

# Turn 2 — test if it remembers
ask_agent("What is my name and where do I work?")

# Turn 3 — search for policy
ask_agent("Search for the refund policy")

# Turn 4 — follow up on previous search
ask_agent("Summarize what you just found and tell me how many days I have to return a product")

# Turn 5 — combine memory + tool
ask_agent("If I return a product today what is the last date I can expect my refund? Use the refund processing time.")

# ─────────────────────────────────────────
# SHOW CONVERSATION HISTORY
# ─────────────────────────────────────────

print(f"\n{'='*60}")
print(f"Total messages in memory: {len(conversation_history)}")
for i, msg in enumerate(conversation_history):
    print(f"\n[{i+1}] {msg['role'].upper()}: {msg['content'][:100]}...")
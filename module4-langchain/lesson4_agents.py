import os
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool

load_dotenv()

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# ─────────────────────────────────────────
# TOOLS
# ─────────────────────────────────────────

@tool
def calculate(expression: str) -> str:
    """
    Useful for doing math calculations.
    Input should be a valid Python math expression.
    Example: '2 + 2' or '100 * 1.18' or '28 * 9/5 + 32'
    """
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def get_current_date(input: str = "") -> str:
    """
    Returns the current date and time.
    Use this when user asks about today's date or current time.
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def search_company_policy(query: str) -> str:
    """
    Search company policies and documents.
    Use this when user asks about refund, leave, travel, security or tech stack policies.
    Input should be the topic to search for.
    """
    policies = {
        "refund": "Customers can return products within 30 days for a full refund. Refunds processed in 5-7 business days.",
        "leave": "Full-time employees get 20 days paid annual leave. Manager approval needed 2 weeks in advance.",
        "travel": "Accommodation up to 5000 rupees per night. Meals up to 1000 rupees per day.",
        "security": "Passwords must be 12+ characters. Changed every 90 days. Two-factor authentication mandatory.",
        "tech": "Backend uses FastAPI, PostgreSQL, Redis. All services containerized with Docker."
    }

    query_lower = query.lower()
    for key, value in policies.items():
        if key in query_lower:
            return value

    return "No specific policy found for this query."


# ─────────────────────────────────────────
# AGENT SETUP — LangChain 1.4.x new API
# ─────────────────────────────────────────

tools = [calculate, get_current_date, search_company_policy]

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="You are a helpful assistant. Use the available tools to answer questions accurately."
)

# ─────────────────────────────────────────
# TEST THE AGENT
# ─────────────────────────────────────────

def ask_agent(question: str):
    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print('='*60)
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    # Get the last message which is the final answer
    final_message = result["messages"][-1]
    print(f"\nFinal Answer: {final_message.content}")


ask_agent("What is 15% of 85000?")
ask_agent("What is today's date?")
ask_agent("What is the refund policy?")
ask_agent("What is today's date and what is the travel expense limit per day?")
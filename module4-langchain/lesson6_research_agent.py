import os
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_tavily import TavilySearch

load_dotenv()

llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

#TOOL 1 - Web search using Tavily

search_tool=TavilySearch(
    max_results=3,
    topic="general"
)

#TOOL 2 - Save research to file
@tool
def save_research(content:str)->str:
    """
    Save research findings to a file.
    Use this when you have completed your research and want to save the summary.
    Input should be the complete research summary to save.
    """

    timestamp=datetime.now().strftime("%y%m%d_%H%M%S")
    filename=f"research_{timestamp}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"Research Report\n")
        f.write(f"Generated: {datetime.now().strftime('%y-%m-%d %H:%M:%S')}\n")
        f.write("="*50 + "\n\n")
        f.write(content)
    return f"research saved to {filename}"

#TOOL 3 - Get current date
@tool
def get_current_date()->str:
    """
    Returns the current date and time.
    Use when research needs to be timestamped or dated.
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

#RESEARCH AGENT
tools=[search_tool,save_research,get_current_date]

agent=create_agent(
    model=llm,
    tools=tools,
    system_prompt="""You are an expert research agent.
when given a research topic:
1.Search for relevant and current information
2.Analyze and synthesize what you find
3.Create a comprehenssive but concise summary
4.Save the research findings to a file

Always cite your sources. Always save your findings."""
)

#RUN RESEARCH
def research(topic:str):
    print(f"\n{'='*60}")
    print(f"Research Topic:{topic}")
    print('='*60)
    result=agent.invoke({
        "messages":[{
            "role":"user",
            "content":f"Reseasrch this topic and save your findings :{topic}"
        }]
    })
    final_answer=result["messages"][-1].content
    print(f"\nAgent Report:\n{final_answer}")


#Run it
research("Lates best practices for RAG systems in production 2026")
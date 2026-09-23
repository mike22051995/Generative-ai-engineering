import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)


#MEMORY
chat_history= []

#PROMPT - includes chat history placeholder

prompt=ChatPromptTemplate.from_messages([
    ("system","you are a helpful assistant."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

#CHAIN
chain=llm | llm

#CHAT FUNCTION -  manages history manually

def chat(user_input:str)->str:
    messages = [
        ("system", "You are a helpful assistant."),
        *[(msg.__class__.__name__.replace("Message", "").lower(), msg.content) 
          for msg in chat_history],
        ("human", user_input)
    ]
    
    response = llm.invoke(messages)
    
    # Save to history
    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=response.content))
    
    return response.content

print("=== Turn 1 ===")
print(f"AI: {chat('My name is Mukesh and I work at Nutanix')}\n")

print("=== Turn 2 ===")
print(f"AI: {chat('What is my name?')}\n")

print("=== Turn 3 ===")
print(f"AI: {chat('Where do I work?')}\n")

print("=== Turn 4 ===")
print(f"AI: {chat('What have we talked about so far?')}\n")

# ─────────────────────────────────────────
# SEE WHAT MEMORY STORED
# ─────────────────────────────────────────

print("=== Memory Contents ===")
for msg in chat_history:
    print(f"{msg.__class__.__name__}: {msg.content}")


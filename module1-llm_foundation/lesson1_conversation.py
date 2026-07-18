import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


client=OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))



def chat_With_summary():
    '''a simple multiturn conversation.
    we manually maintain the message.
    watch how it works'''
    message=[
        {"role":"system","content":"you are a helpful assistent, answer concisely"}
        ]
    
    print(f"chat started : type 'quit' to end conversation.\n ")
    while True:
        user_input=input("you   ").strip()
        if user_input.lower()=="quit":
            break

        message.append({"role":"user", "content":user_input})
        response=client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=message,
            temperature=0.7,
            max_tokens=150
        )
        assistant_reply=response.choices[0].message.content
        message.append({"role":"assistant", "content":assistant_reply})
        print(f"assistant: {assistant_reply}")
        print(f"[tokens this call:{response.usage.total_tokens}]\n ")

chat_With_summary()

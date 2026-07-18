import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

client=OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def compare_prompt(wek:str,strong:str,context:str):
    '''compare weak vs strong prompt on same task to demonstrate the power of prompting'''
    for lable,prompt in [("WEAK",weak),("STRONG",strong)]:
        response=client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role":"system", "content":prompt},
                {"role":"user","content":context}
            ],
            temperature=0.3,
            max_tokens=300
            )
        print(f"\n -----{lable}---PROMPT OUTPUT")
        print(f"\n {response.choices[0].message.content}")
        print(f"tokens used={response.usage.total_tokens}")


weak="review this code"
strong="""you are a senior Python engineer specialising in FastAPI.
Review the code below and provide feedback in exactly this format:
BUGS:(list any bugs, or 'NONE FOUND')
PERFORMANCE:(list improvements)
READABILITY:(list improvement)"""

code="""
def get_user(user_id):
    conn=psycopg2.connect("postgresql://localhost/mydb)
    cursor=conn.cursor()
    cursor.execute(f"SELECT * FROM users WHERE id=
    {user_id}")
    return cursor.fetchone()
    """

compare_prompt(weak,strong,code)
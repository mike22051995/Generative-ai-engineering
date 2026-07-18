import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client=OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def analyze_code_review(context:str):
    '''anyze code review and produces structured output'''
    system_prompt="""
You are a senior Python engineer specializing in Python code review.
return the answer in the valid JSON object.
do not return markdown.
do not return explanations.
Return only JSON.
Review the code below and answer exaclty in the below format:
{
"bugs":["bugs1","bugs2"],
"security_issues":["issues1","issues2"],
"performance":["improvement1"],
"overall_rating":"need_improvement" | "good" | "critical"
}
Example output for bad code:
{
    "bugs": ["connection never closed", "missing import for psycopg2"],
    "security_issues": ["SQL injection via f-string formatting"],
    "performance": ["use connection pooling instead of new connection per call"],
    "overall_rating": "critical"
    }


"""
    response=client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role":"system", "content":system_prompt},
            {"role":"user", "content":context},
            
        ],
        response_format={"type":"json_object"},
        temperature=0.3
    )
    raw=response.choices[0].message.content
    data=json.loads(raw)

    print(f"\n ------code review output---------")
    print(f"\n🐛 BUGS ({len(data['bugs'])} found):")
    for bug in data["bugs"]:
        print(f"     -{bug}")
    print(f"\n 🛅 SECURITY ISSUES ({len(data["security_issues"])}) found")
    for issue in data["security_issues"]:
        print(f"    -{issue}")
    print(f"\n ⚡PERFORMANCE ISSUE ({len(data["performance"])}) found")

    for performance in data["performance"]:
        print(f"      -{performance}")
    

    print(f"\n OVERALL RATING: ({(data["overall_rating"]).upper()})")

    print(f"\n Tokens used: {response.usage.total_tokens}")
    print("================================")

    return data

code="""
def get_user(user_id):
    conn = psycopg2.connect("postgresql://localhost/mydb")
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM users WHERE id = {user_id}")
    return cursor.fetchone()
"""

analyze_code_review(code)


  
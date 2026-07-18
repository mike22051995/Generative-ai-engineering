import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client=OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# response=client.chat.completions.create(
#     model="gpt-3.5-turbo",
#     messages=[
#         {"role":"system", 
#          "content":"you are a helful assistant"},
#         {"role":"user",
#          "content":"what is the capital of India?"}
#     ],
#     temperature=0.7,
#     max_tokens=100,
#     top_p=1,
#     n=1
# )
temperature=[0.0, 0.7,1.5]
def compare_temperature(prompt:str):
    '''run the same prompt at 3 different temperature.
    see how the output is changing'''
    for temp in temperature:
        response=client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role":"system", "content":"you are helpful assistant"},
                    {"role":"user", "content":prompt}
                ],
                    temperature=temp,
                    max_tokens=60,
                

                )
        print(f"\n --------temperature :{temp}-------")
        print(response.choices[0].message.content)
        print(f"finish reason:",response.choices[0].finish_reason)
        print(f"total tokens used:",response.usage.total_tokens)

compare_temperature("write one sentence about ocean")
        





import tiktoken

enc=tiktoken.get_encoding("cl100k_base")


def show_token(text:str):
    '''show exactly how string gets broken into token
    then count of the token'''

    token_id=enc.encode(text)
    token_chunk=[enc.decode([tid]) for tid in token_id]

    print(f"\n Text: {repr(text)}")
    print(f"\n Token ids: {token_id}")
    print(f"\n Token chunks: {token_chunk}")
    print(f"\n Token count: {len(token_id)}")


# show_token("hello")
# show_token("   hello")
show_token("what is the capital of India")
show_token("फ्रांस की राजधानी पेरिस है")
# show_token(" def add(a:int,b:int)->int")
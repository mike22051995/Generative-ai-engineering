import torch
from transformers import AutoTokenizer, AutoModel

MODEL_NAME="bert-base-uncased"
tokenizer=AutoTokenizer.from_pretrained(MODEL_NAME)
model=AutoModel.from_pretrained(
    MODEL_NAME,
    output_attentions=True
)
model.eval()


def show_attention(sentence:str):
    '''Tokenize a sentence and show what each token
    attends to most strongly.'''
    

    inputs=tokenizer(sentence,return_tensors="pt")
    tokens=tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    print(f"\nSentence: {sentence}")
    print(f"Tokens: {tokens}")
    print(f"Token count:{len(tokens)}")

    with torch.no_grad():
        outputs=model(**inputs)
    attentions=outputs.attentions
    print(f"\n nummber of tranformers layers: {len(attentions)}")
    print(f"number of attention heads per layer: {attentions[0].shape[1]}")

    last_layer=attentions[-1]
    avg_attention=last_layer[0].mean(dim=0)
    print(f"\n Attention from each token (What it looks at most)------")
    

    for i, token in enumerate(tokens):
        attention_row=avg_attention[i]
        max_idx=attention_row.argmax().item()
        max_score=attention_row[max_idx].item()
        print(f"  '{token}' → attends most to '{tokens[max_idx]}' "
              f"(score: {max_score:.3f})")
        
show_attention("The animal didn't cross the street because it was tired")



show_attention("Mukesh is a backend engineer who loves Python")



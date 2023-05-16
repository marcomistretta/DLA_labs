from transformers import GPT2Tokenizer, pipeline

train_txt = "dante_train.txt"

with open(train_txt, 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
txt_lenght = len(text)
tokenized_txt_lenght = len(tokenizer(text)['input_ids'])
print("numero caratteri train divina commedia:", txt_lenght)
print("lenght tokenized divina commedia:", tokenized_txt_lenght)
print("ratio:", tokenized_txt_lenght / txt_lenght)

generator = pipeline('text-generation', model='gpt2')
input_prompt = "Halfway down the road of life"
print("input text:", input_prompt)
# print(generator(input_prompt, max_length=50, num_return_sequences=1, temperature=0.9))
print("generator:", generator(input_prompt, max_length=50, num_return_sequences=1))

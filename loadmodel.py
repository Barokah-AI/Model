from transformers import BertTokenizer, BertForSequenceClassification
import torch
import pandas as pd

version = 10

df = pd.read_csv("dataset/barokah.csv", sep="|")

# Encode labels
df['label'] = df['answer'].astype('category').cat.codes
label_dict = dict(enumerate(df['answer'].astype('category').cat.categories))

tokenizer = BertTokenizer.from_pretrained("./models/model v"+str(version))
model = BertForSequenceClassification.from_pretrained("./models/model v"+str(version))

def get_answer(question):
    input_s = tokenizer(question, return_tensors="pt")
    input_s = question
    output = model(**input_s)
    answer = torch.argmax(output.logits).item()
    print(max(output.logits))
    print(answer)
    return label_dict[answer]
    # return output

while True:
    user_input = input("Tanyakan sesuatu (atau ketik 'exit' untuk keluar): ")
    if user_input.lower() == 'exit':
        print("Terima kasih! Sampai jumpa!")
        break
    answer = get_answer(user_input)
    print(f"Jawaban: {answer}")

# print(label_dict)
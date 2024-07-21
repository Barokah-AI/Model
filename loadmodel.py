from transformers import BertTokenizer, BertForSequenceClassification
import torch
import pandas as pd

version = 6

df = pd.read_csv("dataset/13002-14001.csv", sep="|")

# Encode labels
df['label'] = df['answer'].astype('category').cat.codes
label_dict = dict(enumerate(df['answer'].astype('category').cat.categories))


tokenizer = BertTokenizer.from_pretrained("./models/model v"+str(version))
model = BertForSequenceClassification.from_pretrained("./models/model v"+str(version))


def get_answer(question):
    inputs = tokenizer(question, return_tensors="pt")
    outputs = model(**inputs)
    answer = torch.argmax(outputs.logits, dim=1).item()  # Ensure the correct dimension is specified
    print(torch.max(outputs.logits, dim=1))  # Print the max value for each logit
    print(answer)
    return label_dict[answer]

    # return outputs

# Test the model
while True:
    user_input = input("Tanyakan sesuatu (atau ketik 'exit' untuk keluar): ")
    if user_input.lower() == 'exit':
        print("Terima kasih! Sampai jumpa!")
        break
    answer = get_answer(user_input)
    print(f"Jawaban: {answer}")

print(label_dict)
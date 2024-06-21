from transformers import BertTokenizer, BertForSequenceClassification
import torch
import pandas as pd

df = pd.read_csv("dataset/questions.csv", sep="|")

# Encode labels
df['label'] = df['answer'].astype('category').cat.codes
label_dict = dict(enumerate(df['answer'].astype('category').cat.categories))


tokenizer = BertTokenizer.from_pretrained("./qa_model4")
model = BertForSequenceClassification.from_pretrained("./qa_model4")


def get_answer(question):
    inputs = tokenizer(question, return_tensors="pt")
    outputs = model(**inputs)
    answer = torch.argmax(outputs.logits).item()
    return label_dict[answer]

# Test the model
while True:
    user_input = input("Tanyakan sesuatu (atau ketik 'exit' untuk keluar): ")
    if user_input.lower() == 'exit':
        print("Terima kasih! Sampai jumpa!")
        break
    answer = get_answer(user_input)
    print(f"Jawaban: {answer}")
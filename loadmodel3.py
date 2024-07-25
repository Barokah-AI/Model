# Use a pipeline as a high-level helper
from transformers import pipeline
import torch
import pandas as pd

pipe = pipeline("text-classification", model="dani3390/barokah-ai")

# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("dani3390/barokah-ai")
model = AutoModelForSequenceClassification.from_pretrained("dani3390/barokah-ai")


df = pd.read_csv("dataset/barokah.csv", sep="|")
# Encode labels
df['label'] = df['answer'].astype('category').cat.codes
label_dict = dict(enumerate(df['answer'].astype('category').cat.categories))
def get_answer(question):
    inputs = tokenizer(question, return_tensors="pt")
    outputs = model(**inputs)
    answer = torch.argmax(outputs.logits).item()
    # print(max(outputs.logits))
    # print(answer)
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

# print(label_dict)
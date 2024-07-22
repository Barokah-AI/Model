# Description: Load the model and tokenizer from the specified version and test it with user input
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import pandas as pd

# Load the model and tokenizer
version = 6

# Load dataset
df = pd.read_csv("dataset/13002-14001.csv", sep="|")

# Encode labels
df['label'] = df['answer'].astype('category').cat.codes
label_dict = dict(enumerate(df['answer'].astype('category').cat.categories))

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained("./models/model v"+str(version))
model = BertForSequenceClassification.from_pretrained("./models/model v"+str(version))

# Tokenize dataset and include labels
def get_answer(question):
    inputs = tokenizer(question, return_tensors="pt")
    outputs = model(**inputs)
    answer = torch.argmax(outputs.logits).item()
    print(max(outputs.logits))
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
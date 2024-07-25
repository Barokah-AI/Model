import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd

#pilih versi model yang akan digunakan
version = 12
# Load dataset
df = pd.read_csv("dataset/barokah.csv", sep="|")

# Encode labels
df['label'] = df['answer'].astype('category').cat.codes

# Buat dictionary dari label
label_dict = dict(enumerate(df['answer'].astype('category').cat.categories))

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained("./models/model v"+str(version))

# Load model
model = BertForSequenceClassification.from_pretrained("./models/model v"+str(version))

# Fungsi untuk mendapatkan jawaban dari pertanyaan
def get_answer(question):
    # Tokenisasi pertanyaan
    input_s = tokenizer(question, return_tensors="pt")
    # Prediksi jawaban
    input_s = question
    # Prediksi jawaban dari pertanyaan
    output = model(**input_s)
    # Ambil label dengan nilai tertinggi
    answer = torch.argmax(output.logits).item()
    return label_dict[answer]

while True:
    user_input = input("Tanyakan sesuatu (atau ketik 'exit' untuk keluar): ")
    if user_input.lower() == 'exit':
        print("Terima kasih! Sampai jumpa!")
        break
    answer = get_answer(user_input)
    print(f"Jawaban: {answer}")
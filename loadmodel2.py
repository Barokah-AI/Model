# Load model T5 yang sudah di-training sebelumnya dan gunakan untuk menghasilkan jawaban dari pertanyaan yang diberikan
from transformers import T5Tokenizer, T5ForConditionalGeneration
# transformers adalah library yang menyediakan berbagai model NLP yang sudah di-training sebelumnya

# Load model dan tokenizer dari direktori tempat Anda menyimpannya
model_path = 't5_qa_model'
#  Load model dari direktori yang sudah di-training sebelumnya
model = T5ForConditionalGeneration.from_pretrained(model_path)
# Load tokenizer dari direktori yang sudah di-training sebelumnya
tokenizer = T5Tokenizer.from_pretrained(model_path)
# Tanyakan pertanyaan
question = "Dimana kita bisa makan siang?"
# Tokenisasi pertanyaan
inputs = tokenizer.encode("generate answer: " + question, return_tensors="pt", max_length=64, truncation=True)
# Prediksi jawaban
outputs = model.generate(inputs, max_length=50, num_return_sequences=1, early_stopping=True)
answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Answer:", answer)

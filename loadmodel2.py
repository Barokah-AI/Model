from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load model dan tokenizer dari direktori tempat Anda menyimpannya
model_path = 't5_qa_model'
model = T5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = T5Tokenizer.from_pretrained(model_path)

question = "Dimana kita bisa makan siang?"
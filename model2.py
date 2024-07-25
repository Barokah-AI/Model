import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
import torch

# Load the dataset
df = pd.read_csv('dataset/questions.csv', delimiter='|', names=['question', 'answer'])

# Prepare the dataset
tokenizer = T5Tokenizer.from_pretrained('t5-small')
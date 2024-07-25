import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
import torch

# Load the dataset
df = pd.read_csv('dataset/questions.csv', delimiter='|', names=['question', 'answer'])

# Prepare the dataset
tokenizer = T5Tokenizer.from_pretrained('t5-small')

# Combine question and answer into a single string for training
inputs = "generate answer: " + df['question'] + " </s>"
targets = df['answer'] + " </s>"

class QADataset(torch.utils.data.Dataset):
  def __init__(self, inputs, targets, tokenizer, max_length=64):
    self.inputs = inputs
    self.targets = targets
    self.tokenizer = tokenizer
    self.max_length = max_length

  def __len__(self):
    return len(self.inputs)
  
  def __getitem__(self, idx):
    input_encodings = self.tokenizer(
        self.inputs[idx], truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")
    target_encodings = self.tokenizer(
      self.targets[idx], truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")
    
    input_ids = input_encodings['input_ids'].squeeze()
    attention_mask = input_encodings['attention_mask'].squeeze()
    labels = target_encodings['input_ids'].squeeze()
  

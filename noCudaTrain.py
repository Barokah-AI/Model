import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
import os
import re
import evaluate
import numpy as np

models_directory = "./models"
name_models = "model v"

def get_next_model_version(models_directory):
    contents = os.listdir(models_directory)
    max_version = 0
    for item in contents:
        match = re.search(r'v(\d+)', item)
        if match:
            number = int(match.group(1))
            max_version = max(max_version, number)
    return name_models + str(max_version + 1)


# Load dataset
df = pd.read_csv("dataset/barokah-1.csv", sep="|")

# Encode labels
df['label'] = df['answer'].astype('category').cat.codes
label_dict = dict(enumerate(df['answer'].astype('category').cat.categories))

# Split dataset
train_df, eval_df = train_test_split(df, test_size=0.2)

# Convert to Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_df)
eval_dataset = Dataset.from_pandas(eval_df)

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(df['label'].unique()))

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

# Verify CUDA availability and device
print("CUDA available:", torch.cuda.is_available())
print("Number of GPUs:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("CUDA device name:", torch.cuda.get_device_name(0))
else:
    print("CUDA not available. Check your CUDA installation and NVIDIA drivers.")

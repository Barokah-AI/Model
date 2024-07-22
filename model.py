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

# Load dataset
df = pd.read_csv("dataset/barokah.csv", sep="|")

# dataset di kali 5
df = pd.concat([df]*5, ignore_index=True)

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

# Tokenize dataset and include labels
def preprocess_function(examples):
    inputs = tokenizer(examples['question'], truncation=True, padding='max_length', max_length=128)
    inputs['label'] = examples['label']
    return inputs

train_dataset = train_dataset.map(preprocess_function, batched=True)
eval_dataset = eval_dataset.map(preprocess_function, batched=True)

# Set format for PyTorch
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
eval_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# Validate labels
num_labels = len(df['label'].unique())
for dataset in [train_dataset, eval_dataset]:
    for example in dataset:
        assert 0 <= example['label'] < num_labels, f"Invalid label {example['label']} found!"

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=40,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    warmup_steps=8000,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=400,
    evaluation_strategy="steps",
    learning_rate=3e-5,
    save_total_limit=5,
    disable_tqdm=False,  # Set to True if you don't want to use tqdm progress bars
    load_best_model_at_end=True,
    metric_for_best_model="eval_accuracy",
    greater_is_better=True,
    no_cuda=not torch.cuda.is_available()  # Ini akan menggunakan GPU jika tersedia
)

# Explicitly move model to GPU
if torch.cuda.is_available():
    model.to(torch.device("cuda"))
    print("Model moved to GPU")
else:
    print("CUDA not available. Training on CPU.")

# Define accuracy metric
metric = evaluate.load("accuracy", trust_remote_code=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, np.ndarray):
        logits = torch.tensor(logits)
    if isinstance(labels, np.ndarray):
        labels = torch.tensor(labels)
    predictions = torch.argmax(logits, dim=-1)
    return metric.compute(predictions=predictions, references=labels)


# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics
)
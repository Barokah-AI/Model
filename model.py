from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from datasets import Dataset
import evaluate
import numpy as np

# Verify CUDA availability and device
print("CUDA available:", torch.cuda.is_available())
print("Number of GPUs:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("CUDA device name:", torch.cuda.get_device_name(0))
else:
    print("CUDA not available. Check your CUDA installation and NVIDIA drivers.")

# Load dataset
df = pd.read_csv("dataset/questions.csv", sep="|")

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
    inputs = tokenizer(examples['question'], truncation=True, padding=True)
    inputs['label'] = examples['label']
    return inputs

train_dataset = train_dataset.map(preprocess_function, batched=True)
eval_dataset = eval_dataset.map(preprocess_function, batched=True)

# Set format for PyTorch
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
eval_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir='./logs',
    # Menambahkan parameter untuk penggunaan GPU
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
    # Convert logits to tensor if it is a numpy array
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

# Print device to verify
print("Training on device:", trainer.args.device)


# Train model
trainer.train()


# Evaluate model
eval_results = trainer.evaluate()

# Print evaluation results, including accuracy
print(f"Evaluation results: {eval_results}")

# Save model
model.save_pretrained("./results")
tokenizer.save_pretrained("./results")
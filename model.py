from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from datasets import Dataset

# Load dataset
df = pd.read_csv("dataset/questions.csv", sep="|")

# Encode labels
df['label'] = df['answer'].astype('category').cat.codes
label_dict = dict(enumerate(df['answer'].astype('category').cat.categories))

# Split dataset
train_df, eval_df = train_test_split(df, test_size=0.1)

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
    no_cuda=False  # Ini akan menggunakan GPU jika tersedia
)

# Trainer
trainer = Trainer(
    model=model,                         
    args=training_args,                  
    train_dataset=train_dataset,         
    eval_dataset=eval_dataset             
)

# Train model
trainer.train()


# Evaluate model
eval_results = trainer.evaluate()

# Print evaluation results
print(eval_results)

# Save model
model.save_pretrained("./qa_model6")
tokenizer.save_pretrained("./qa_model6")
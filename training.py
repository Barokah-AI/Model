import tensorflow as tf_keras
from transformers import TFBertForSequenceClassification, BertTokenizer
import pandas as pd
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('questions.csv', sep='|', usecols=['question', 'answer'])

# Preprocessing functions
factory = StemmerFactory()
stemmer = factory.create_stemmer()

punct_re_escape = re.compile('[%s]' % re.escape('!"#$%&()*+,./:;<=>?@[\\]^_`{|}~'))

def normalize_sentence(sentence):
    sentence = punct_re_escape.sub('', sentence.lower())
    sentence = ' '.join(sentence.split())
    if sentence:
        sentence = sentence.strip().split(" ")
        normal_sentence = " "
        for word in sentence:
            root_sentence = stemmer.stem(word)
            normal_sentence += root_sentence + " "
        return punct_re_escape.sub('', normal_sentence.strip())
    return sentence

# Clean and preprocess the dataset
cleaned_data = []
for index, row in df.iterrows():
    question = normalize_sentence(str(row['question']))
    answer = str(row['answer']).lower().replace('\n', ' ')

    if len(question.split()) > 0:
        cleaned_data.append({"question": question, "answer": answer})

df_cleaned = pd.DataFrame(cleaned_data)

# Tokenisasi dan persiapan data
tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p2')
input_ids = []
attention_masks = []

for index, row in df_cleaned.iterrows():
    encoded = tokenizer.encode_plus(
        row['question'],
        add_special_tokens=True,
        max_length=64,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='tf'
    )
    input_ids.append(encoded['input_ids'])
    attention_masks.append(encoded['attention_mask'])

input_ids = tf.concat(input_ids, axis=0)
attention_masks = tf.concat(attention_masks, axis=0)
labels = tf.constant(df_cleaned['answer'].values)

# Split data menjadi train dan test set
train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels, test_size=0.1)

# Konversi ke TensorFlow Dataset
train_dataset = tf.data.Dataset.from_tensor_slices(({'input_ids': train_inputs, 'attention_mask': attention_masks[:len(train_inputs)]}, train_labels))
validation_dataset = tf.data.Dataset.from_tensor_slices(({'input_ids': validation_inputs, 'attention_mask': attention_masks[len(train_inputs):]}, validation_labels))

# Load model
model = TFBertForSequenceClassification.from_pretrained('indobenchmark/indobert-base-p2', num_labels=2)

# Compile model
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

# Train model
epochs = 1
batch_size = 16

history = model.fit(train_dataset.shuffle(100).batch(batch_size), epochs=epochs, batch_size=batch_size, validation_data=validation_dataset.batch(batch_size))


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


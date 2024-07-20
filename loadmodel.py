from transformers import BertTokenizer, BertForSequenceClassification
import torch
import pandas as pd

version = 6

df = pd.read_csv("dataset/13002-14001.csv", sep="|")
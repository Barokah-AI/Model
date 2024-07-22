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
import pandas as pd
# Load dataset
# Load dataset
df = pd.read_csv("dataset/barokah.csv", sep="|")

# Encode labels
df['label'] = df['answer'].astype('category').cat.codes
label_dict = dict(enumerate(df['answer'].astype('category').cat.categories))
print("Label dictionary:", label_dict)


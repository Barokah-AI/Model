import requests
from transformers import BertTokenizer

API_URL = "https://api-inference.huggingface.co/models/dani3390/barokah-ai"
headers = {"Authorization": "Bearer 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"}


tokenizer = BertTokenizer.from_pretrained("./models/model v"+str(12))



def query(payload):
    # Tokenize the input text
    inputs = tokenizer(payload["inputs"], return_tensors="pt")

    # Send the request with tokenized inputs
    response = requests.post(API_URL, headers=headers, json=inputs)
    return response.json()

output = query({
	"inputs": "Apa langkah pertama dalam menangani pesan kesalahan",
})

print(output[0][0]['label'])
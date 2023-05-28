# 用于测试模型输出
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load trained model and tokenizer
path = "./models/checkpoint-30000/"
tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
model = AutoModelForSeq2SeqLM.from_pretrained(path).to(device)

print('test')
while True:
    text = input('Sentence:')
    inputs = [text.replace(' ', '')]

    # Tokenize and prepare the inputs for model
    input_ids = tokenizer(inputs, return_tensors="pt", max_length=512,
                          truncation=True, padding="max_length").input_ids.to(device)
    attention_mask = tokenizer(inputs, return_tensors="pt", max_length=512,
                               truncation=True, padding="max_length").attention_mask.to(device)

    # Generate prediction
    output = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=512)

    # Decode the prediction
    decoded_output = [tokenizer.decode(ids, skip_special_tokens=True) for ids in output]

    for sentence in decoded_output:
        print(f"Predict : {sentence}")
        print('*' * 100)
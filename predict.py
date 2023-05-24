# 用于测试模型输出
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import string
import re
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("./models/checkpoint-20000/").to(device)

while True:
    test_data = [input('Sentence:')]
    # test_data = ["你好 我是李沐", "都什么年代了 还在抽传统香烟", '不许你说丁真 他是我的偶像',
    #              '我们的价值观是 富强 民主 文明 和谐', '这是一个句子 这个句子在此结束']
    inputs = [re.sub(r'\s+', ' ', x) for x in test_data]

    # Tokenize and prepare the inputs for model
    input_ids = tokenizer(inputs, return_tensors="pt", max_length=512, truncation=True, padding="max_length").input_ids.to(device)
    attention_mask = tokenizer(inputs, return_tensors="pt", max_length=512, truncation=True, padding="max_length").attention_mask.to(device)

    # Generate prediction
    output = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=512)

    # Decode the prediction
    decoded_output = [tokenizer.decode(ids, skip_special_tokens=True) for ids in output]

    for i, sentence in enumerate(decoded_output):
        # print(f"Sentence: {test_data[i]}")
        print(f"Predict : {sentence}")
        print('*' * 100)
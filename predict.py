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
    # test_data = ["猪笼草原产于热带和亚热带地区 现主要分布在东南亚一带 中国广东 广西等地有分布 猪笼草喜欢湿润和温暖半阴的生长环境 不耐寒 怕积水 怕强光 怕干燥 喜欢疏松 肥沃和透气的腐叶土和泥炭土 对光照要求较为严格 猪笼草的繁殖方式包括扦插繁殖 压条繁殖和播种繁殖"]
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
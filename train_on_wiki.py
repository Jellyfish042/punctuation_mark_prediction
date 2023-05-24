import datasets
import string
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments
import re
import numpy as np
from datasets import Dataset

# os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'


print('train test begin 007')


def replace_punctuation_with_space(s):
    # 创建一个正则表达式模式，匹配所有标点符号
    pattern = '[' + string.punctuation + '。，“”‘’！？【】：；……、（）《》·「」]'
    # 将标点符号替换为一个空格，并保留换行符
    return re.sub(pattern, ' ', s)


def preprocess_function(examples):
    inputs = [replace_punctuation_with_space(x) for x in examples['text']]
    # inputs = ['Add punctuation mark to the following text:' + re.sub(r'\s+', ' ', x) for x in inputs]
    inputs = [re.sub(r'\s+', ' ', x) for x in inputs]
    targets = [x for x in examples['text']]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding='max_length')

    # 需要确保模型输入包含labels
    model_inputs["labels"] = tokenizer(targets, max_length=512, truncation=True, padding='max_length').input_ids

    return model_inputs


def data_split(ds, val_split=0.1):
    # 获取数据集的长度
    len_dataset = len(ds)

    # 创建一个包含数据集所有索引的数组
    indices = np.arange(len_dataset)

    # 打乱索引
    np.random.shuffle(indices)

    # 确定训练集和验证集的划分点
    split = int(np.floor(val_split * len_dataset))

    # 根据划分点划分训练集和验证集的索引
    val_indices, train_indices = indices[:split], indices[split:]

    # 使用划分的索引创建训练集和验证集
    train_ds = ds.select(train_indices)
    val_ds = ds.select(val_indices)
    return train_ds, val_ds


tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")

ds = datasets.load_dataset('wikipedia', date='20230301', language='zh', beam_runner='DirectRunner')
ds = ds['train']
ds = ds.map(preprocess_function, batched=True)
train_ds, val_ds = data_split(ds)

# 为数据集应用预处理
# train_ds = train_ds.map(preprocess_function, batched=True)
# val_ds = val_ds.map(preprocess_function, batched=True)

model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-small")

# 定义训练参数
training_args = Seq2SeqTrainingArguments(
    output_dir="./models",
    evaluation_strategy="steps",
    eval_steps=4000,
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=5,
    weight_decay=0.01,
    gradient_accumulation_steps=16,
    save_strategy="steps",
    save_steps=2500,
    save_total_limit=20,
    logging_steps=200,
)

print("Training configuration:")
print(training_args)

# 初始化训练器
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds
)

# 开始训练
trainer.train()

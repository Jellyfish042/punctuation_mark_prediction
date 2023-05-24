# 模型训练
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_from_disk
from support import *

# 加载数据集
dataset = load_from_disk('./datasets/amazon_and_wiki500k')
# 数据集分割
train_ds, val_ds = data_split(dataset)
print(train_ds, val_ds)

# model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-small")  # 下载预训练模型
model = AutoModelForSeq2SeqLM.from_pretrained("./models/checkpoint-22500")  # 从本地checkpoint开始

# 定义训练参数
training_args = Seq2SeqTrainingArguments(
    output_dir="./models",
    evaluation_strategy="steps",
    eval_steps=5000,
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
    fp16=True,
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

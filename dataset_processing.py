# 处理数据集
import datasets
import string
from transformers import AutoTokenizer
from tqdm import tqdm
import random
import re


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


# 构造长度为1-13个句子的随机长度段落
def construct_random_sentences(ls):
    r = random.randint(1, 13)
    l, u = 0, 0
    ls_len = len(ls)
    results = []
    while u < ls_len:
        l = u
        u += random.randint(1, 13)
        results.append(''.join(ls[l: u]))
    return results


print('train test begin 007')


tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")

# 加载数据集
wiki = datasets.load_dataset('wikipedia', date='20230301', language='zh', beam_runner='DirectRunner')['train']
amazon = datasets.load_dataset('amazon_reviews_multi', 'zh', split='train')

# 切片用于测试
wiki = wiki[:500000]
# amazon = amazon[:100]

# 构造为新的数据集
datalist = []
datalist += wiki['text']
datalist += amazon['review_body']
full_dataset = datasets.Dataset.from_dict({'text': datalist})

random_sentences = []

# 遍历数据集
for example in tqdm(full_dataset['text']):
    # 分割句子
    sentences = re.split(r'([？。！])', example)
    sentences = [sentences[i] + sentences[i+1] for i in range(0, len(sentences)-1, 2)]
    # 过滤重复标点，匿名函数+过滤器
    sentences = list(filter(lambda x: x not in ['？', '！', '。'], sentences))  # 主要用于过滤Amazon数据集中重复的标点
    # 组合为新的随机长度的段落
    s = construct_random_sentences(sentences)
    random_sentences += s

# 将列表转换为一个新的数据集
new_ds = datasets.Dataset.from_dict({'text': random_sentences})

# 预处理，构造训练样本对
new_ds = new_ds.map(preprocess_function, batched=True)
# 保存数据集
file_path = './datasets/amazon_and_wiki500k'
new_ds.save_to_disk(file_path)
print(f'数据集已保存至{file_path}')

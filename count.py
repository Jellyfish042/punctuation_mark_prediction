# 本文件用于统计数据集中每个句子平均有多少个token，根据统计数据更科学地构建训练数据
import datasets
from tqdm import tqdm
from transformers import AutoTokenizer

# 加载Tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")

# 加载数据集
ds = datasets.load_dataset('wikipedia', date='20230301', language='zh', beam_runner='DirectRunner')
# 统计前10000个样本，节约时间
ds = ds['train'][:10000]

sentences = []
total_len = 0
# 以句号为分隔，分割出句子
for example in tqdm(ds['text']):
    splited_example = example.split('。')
    sentences += splited_example
# token化并统计总token数
for example in tqdm(sentences):
    ids = tokenizer(example, return_tensors="pt").input_ids
    total_len += ids.shape[1]
# 总token数除以句子数目，得到平均token数
print(total_len / len(sentences))

# 前10000个样本的统计结果是41.145
# 512 / 41.145 = 12.4
# 构造每个样本由1-12条句子随机构成
# 当然，更严谨的话应该将Amazon数据集的也统计一下，然后按比例加权

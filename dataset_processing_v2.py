# 更好的数据清洗（虽然只改了一点，但是清洗得好得多）
import datasets
import string
from transformers import AutoTokenizer
from tqdm import tqdm
import random
import re


def replace_punctuation_with_space(s):
    '''
    用于替换句中标点
    :param s: 输入字符串
    :return: 无标点字符串
    '''
    # 创建一个正则表达式模式，匹配所有标点符号
    pattern = '[' + string.punctuation + '。，“”‘’！？【】：；……、（）《》·「」 ]'
    # 将标点符号替换为一个空格，并保留换行符
    return re.sub(pattern, ' ', s)


def preprocess_function(examples):
    '''
    用于构造输入和输出，用来做数据集map的函数
    :param examples:
    :return:
    '''
    inputs = [replace_punctuation_with_space(x) for x in examples['text']]
    targets = [x for x in examples['text']]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding='max_length')

    # 需要确保模型输入包含labels
    model_inputs["labels"] = tokenizer(targets, max_length=512, truncation=True, padding='max_length').input_ids

    return model_inputs


def construct_random_sentences(ls):
    '''
    构造长度为1-12个句子的随机长度段落
    :param ls: 列表，包含多个句子
    :return: 随机长度段落
    '''
    l, u = 0, 0
    ls_len = len(ls)
    results = []
    while u < ls_len:
        l = u
        u += random.randint(1, 12)
        results.append(''.join(ls[l: u]))
    return results


def is_contain_chinese(s):
    """
    判断字符串中是否包含中文
    """
    pattern = re.compile(r'[\u4e00-\u9fa5]')
    match = pattern.search(s)
    return match is not None


# 加载Tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")

# 加载数据集
wiki = datasets.load_dataset('wikipedia', date='20230301', language='zh', beam_runner='DirectRunner')['train']
amazon = datasets.load_dataset('amazon_reviews_multi', 'zh', split='train')

# 切片（如果你不想用全部数据的话）
wiki = wiki[:50]
amazon = amazon[:100]

# 将数据集拼成新数据集
datalist = []
datalist += wiki['text']
datalist += amazon['review_body']
full_dataset = datasets.Dataset.from_dict({'text': datalist})

random_sentences = []
# 遍历数据集，做清洗，并且拼成随机长度段落
for example in tqdm(full_dataset['text']):
    # 分割句子
    sentences = re.split(r'([？。！\n])', example)
    sentences = [sentences[i] + sentences[i+1] for i in range(0, len(sentences)-1, 2)]
    # 过滤重复标点，匿名函数+过滤器（独立的标点就是重复的）
    sentences = list(filter(lambda x: x not in ['？', '！', '。', '\n'], sentences))  # 主要用于过滤Amazon数据集中重复的标点
    # 过滤掉不包含中文的句子
    sentences = [x for x in sentences if is_contain_chinese(x)]
    # 过滤掉包含\n的句子，因为我发现这种句子通常质量较低
    sentences = [x for x in sentences if '\n' not in x]
    # 组合为新的随机长度的段落
    s = construct_random_sentences(sentences)
    random_sentences += s

# 将列表转换为一个新的数据集
new_ds = datasets.Dataset.from_dict({'text': random_sentences})

# 预处理，构造训练样本对
new_ds = new_ds.map(preprocess_function, batched=True)
# 保存数据集
file_path = './datasets/amazon_and_wiki_tiny'  # 在此设定数据集的保存路径
new_ds.save_to_disk(file_path)
print(f'数据集已保存至{file_path}')

# 支持函数
import numpy as np


def data_split(ds, val_split=0.1):
    # 获取数据集的长度
    len_dataset = len(ds)

    # 创建一个包含数据集所有索引的数组，并打乱索引
    indices = np.random.permutation(len_dataset)

    # 确定训练集和验证集的划分点
    split = int(np.floor(val_split * len_dataset))

    # 根据划分点划分训练集和验证集的索引
    val_indices, train_indices = indices[:split], indices[split:]

    # 使用划分的索引创建训练集和验证集
    train_ds = ds.select(train_indices)
    val_ds = ds.select(val_indices)
    return train_ds, val_ds

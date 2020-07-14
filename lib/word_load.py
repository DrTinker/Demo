import pandas as pd
from tqdm import tqdm
import pickle
import random

import sys
import os
sys.path.append(os.getcwd())
import config

from lib import is_good_line


# 读取停用词
def load_stop():
    '''
        return: 包含全部停用词的list
    '''
    if not os.path.exists(config.dict_path):
        raise '不存在stopword路径，请检查config配置'
    file = open(config.stopword_path, "r", encoding='utf-8',
                errors='ignore')
    stopwords = []
    stop = file.read()
    for line in stop.split('\n'):
        stopwords.append(line)

    return stopwords


# 把数据读进来
def load_text(filePath, spec='\t', isDF=True, isPrint=True):
    '''
        filePath: 源文件路径
        spec: 问答语句分隔符
        isDF: 是否返回DataFrame格式

        return: 如果isDF为True则返回列名的dataframe
                如果为False怎返回维度为k是问题，v是答案的字典
                e.g: {'你吃饭了吗': ['我没吃呢', '我吃完了'], ......}
    '''
    if not os.path.exists(filePath):
        raise '不存在{}路径，请检查函数调用是否正确'.format(filePath)
    file = open(filePath, "r", encoding='utf-8', errors='ignore')
    lines = file.read()
    raw_data = []

    if isPrint:
        pbar = tqdm()
        for line in lines.split('\n'):
            if(is_good_line(line)):
                raw_data.append(line.split(spec))
                pbar.set_description("正在读取数据：")
                pbar.update()
            else:
                continue
        pbar.close()
    else:
        for line in lines.split('\n'):
            if(is_good_line(line)):
                raw_data.append(line.split(spec))
            else:
                continue

    if isDF:
        # 打乱顺序
        random.shuffle(raw_data)
        raw_data_df = pd.DataFrame(raw_data, columns=['query', 'answer'])
        return raw_data_df
    else:
        d = {}
        for pair in raw_data:
            key = pair[0]
            try:
                value = d[key]
                value.append(pair[1])
            except KeyError:
                value = [pair[1]]
            d[key] = value
        return d


# 读取字典
def load_dict():
    if not os.path.exists(config.dict_file):
        raise '不存在词典路径，请检查config配置'
    print('正在读取字典')
    with open(config.dict_file, mode='rb') as file:
        dictionary = pickle.load(file)

    return dictionary


if __name__ == '__main__':
    res = load_text(config.class_joke, '\t', isDF=False)
    print(res)

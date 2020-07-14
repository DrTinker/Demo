import pickle
import sys
import os
import shutil
sys.path.append(os.getcwd())
import config

from lib import ProgressBar
from lib import ThreadGenerator
from lib import load_dict
from prepare_chat_text import save_data


BATCH_SIZE = config.BATCH_SIZE
# 每100个问答一个batch，每100个batch一个dataset
# 这个变量是多少不用在意，下面的函数中会更新这个值
DATA_LIST_SIZE = 276967
DATA_SET_SIZE = config.DATA_SET_SIZE


# 制作训练集
def get_batch(dictionary, data, batch=BATCH_SIZE):
    print('正在生成dataset')
    matrix = []
    size = 0

    for line in data:
        docs = []
        for word in line:
            try:
                docs.append(dictionary[word])
            except KeyError:
                docs.append(dictionary['<UNK>'])
                continue
        size += 1
        matrix.append(docs)
        if size == batch:
            yield matrix
            # 清空matrix
            matrix = []
            size = 0


def make_dataset():
    # 读入填充PAD UNK完毕的数据
    with open(config.xiaohuangji_pad_que, 'rb') as file:
        que_data = pickle.load(file)
    with open(config.xiaohuangji_pad_ans, 'rb') as file:
        ans_data = pickle.load(file)
    DATA_LIST_SIZE = len(que_data)
    size = DATA_SET_SIZE * BATCH_SIZE
    print(size)
    dictionary = load_dict()

    # 切分in 和 out
    ans_data_in = [data[0:-1] for data in ans_data]
    ans_data_out = [data[1:] for data in ans_data]
    # 删掉原来的dataset
    if os.path.exists(config.dataset_path):
        shutil.rmtree(config.dataset_path)
    # 使用线程生成器，一个getbatch获取50个(batch_size, maxlen)的二维矩阵
    t1 = get_batch(dictionary, que_data, BATCH_SIZE)
    t2 = get_batch(dictionary, ans_data_in, BATCH_SIZE)
    t3 = get_batch(dictionary, ans_data_out, BATCH_SIZE)
    tg1 = ThreadGenerator(t1)
    tg2 = ThreadGenerator(t2)
    tg3 = ThreadGenerator(t3)

    q, a1, a2 = [], [], []

    pbar = ProgressBar(DATA_LIST_SIZE)
    for i in range(0, DATA_LIST_SIZE, BATCH_SIZE):
        # 每100 BATCH一存
        if (i % size == 0 and i != 0) or (DATA_LIST_SIZE-1-i < DATA_SET_SIZE):
            dataset = zip(q, a1, a2)
            save_data(config.dataset_path, '/dataset{}'.
                      format(int(i/size) if i != DATA_LIST_SIZE else
                             int(i/size)+1),
                      dataset)
            # 记得清空list
            q, a1, a2 = [], [], []
            # print("存了")

        q.append(next(tg1))
        a1.append(next(tg2))
        a2.append(next(tg3))
        pbar.update(i)

    tg1.close()
    tg2.close()
    tg3.close()

    return dataset_list


if __name__ == '__main__':
    dataset_list = make_dataset()
    print(len(dataset_list))

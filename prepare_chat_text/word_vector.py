import os
import pickle
import sys
import threading

from gensim import corpora
from gensim.models import Word2Vec
from tqdm import tqdm

sys.path.append(os.getcwd())
import config
from prepare_chat_text import save_data
from lib import load_dict


MIN = config.MIN_FREQUENCE
MAX = config.MAX_FREQUENCE
EMBEDDING_SIZE = config.EMBEDDING_SIZE
KEEP = config.KEEP_WORD
QUE_MAXLEN = config.LEN_OF_QUE
ANS_MAXLEN = config.LEN_OF_ANS
WINDOW = config.WINDOW
MODEL = config.MODE
APPEND = config.APPEND


# 使用gensim生成字典
def make_dict(texts, min_freqency=MIN, max_freqency=MAX, size=None,
              keep=KEEP):
    print('正在生成字典')
    dictionary = corpora.Dictionary(texts)
    docs_num = dictionary.num_docs
    limit = max_freqency / docs_num
    # 设置最大最小词频
    dictionary.filter_extremes(no_below=min_freqency, no_above=limit,
                               keep_n=size, keep_tokens=keep)

    flag = config.ori_dict
    dictionary = dictionary.token2id
    length = len(dictionary)
    for k, v in dictionary.items():
        if v < 4:
            v += length
            dictionary[k] = v
    for k, v in flag.items():
        dictionary[k] = v
    print('字典生成完毕')
    print('字典长度为{}'.format(len(dictionary)))
    return dictionary


# 计算最大长度
def count_maxlen(data):
    maxlen = 0
    for line in data:
        maxlen = maxlen if len(line) < maxlen else len(line)

    return maxlen


# 填充pad和unk
def pad_sequence(data, dictionary, size):
    pbar = tqdm()
    for line in data:
        for j in range(size):
            if j < len(line):
                try:
                    dictionary[line[j]]
                except KeyError:
                    line[j] = '<UNK>'
                    continue
            else:
                line.append('<PAD>')
        if len(line) > size:
            for i in range(size, len(line)):
                line.pop()
        pbar.set_description("正在填充<PAD><UNK>标记：")
        pbar.update()
    pbar.close()


# 训练word2vec模型
def train_word2vec(texts, min_count=MIN, size=EMBEDDING_SIZE,
                   window=WINDOW, mode=MODEL, isAppend=False):
    print('正在训练word2vec模型')
    model = Word2Vec()
    # 先检查是否已经存在模型
    if isAppend and os.path.exists(config.word2vec_model):
        model = Word2Vec.load(config.word2vec_model)
        model.build_vocab(texts, update=isAppend)

    else:
        if not os.path.exists(config.word2vec_path):
            os.makedirs(config.word2vec_path)
        if mode == 'skip-gram':
            model = Word2Vec(min_count=min_count, workers=5,
                             size=size, negative=5, sg=1, hs=1,
                             window=window, sample=0.001)
            model.build_vocab(texts)
        elif mode == 'cbow':
            model = Word2Vec(min_count=min_count, workers=5,
                             size=size, negative=5, sg=0, hs=1,
                             window=window, sample=0.001)
            model.build_vocab(texts)
        else:
            print('模式选择错误')
            return

    model.train(texts, total_examples=model.corpus_count, epochs=model.iter)
    model.save(config.word2vec_model)
    print('训练结束')
    return model


def update_dict(model, isAppend=False):
    # 如果该路径上有词典，而且isAppend为真，就在后面添加
    if os.path.exists(config.dict_file) and isAppend:
        final_dict = load_dict()
    # 如果不存在词典，就新建一个空的词典
    else:
        final_dict = config.ori_dict
    ori_len = len(final_dict)
    print('源字典长度{}'.format(ori_len))
    for k in model.wv.vocab:
        # 先查看有没有这个词，如果没有才增加
        try:
            # 这里默认原本词典的四个标记符也是遵守config里规定的标准的
            final_dict[k]
        except KeyError:
            # 续在后面，所以要加ori_len
            final_dict[k] = ori_len
            ori_len += 1
    print('更新后字典长度为{}'.format(len(final_dict)))
    return final_dict


def step2():
    # 读入数据
    with open(config.xiaohuangji_split_que, 'rb') as file:
        que_data = pickle.load(file)
    with open(config.xiaohuangji_split_ans, 'rb') as file:
        ans_data = pickle.load(file)

    # 问题回答用同一个词表，用来初步筛选低频和高频词汇
    dictionary = make_dict(que_data + ans_data)
    # 增加标签，将不一样长的用PAD填充，高频和低频词使用UNK代替
    # 先填充开始和结束标签，不然会趋向于生成固定长度的句子
    ans_data = [['<SOS>'] + data + ['<EOS>'] for data in ans_data]

    print('问题、回答的最大长度分别为{}、{}'
          .format(count_maxlen(que_data), count_maxlen(ans_data)))
    t1 = threading.Thread(target=pad_sequence, args=(que_data, dictionary,
                          QUE_MAXLEN))
    t2 = threading.Thread(target=pad_sequence, args=(ans_data, dictionary,
                          ANS_MAXLEN))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    # 用填充完PAD的data训练word2vec模型
    model = train_word2vec(que_data + ans_data, isAppend=APPEND)
    # model对高频词汇进行了随机采样，这样会筛选掉一部分原本字典中的词，因此需要在这里进行更新
    final_dict = update_dict(model, isAppend=APPEND)
    save_data(config.dict_path, '/dictionary.txt', final_dict)

    # 保存一下填充网PAD的文本，以便以后使用
    filePath = config.xiaohuangji_pad_path
    fileNames = ['/padding_que.txt', '/padding_ans.txt']
    save_data(filePath, fileNames[0], que_data)
    save_data(filePath, fileNames[1], ans_data)


if __name__ == '__main__':
    step2()

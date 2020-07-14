import os
import sys
import random
sys.path.append(os.getcwd())
import config

from lib import spiltCN_by_word, splitCN_by_char
from lib import load_text
from lib import ProgressBar


BALANCE_NUM = config.BALANCE_NUM
OUTDIR = config.classify_flaged_path


def make_fasttext_corpus(srcPath, outPath, label, blance=0, byword=False):
    '''
        为文本增加标签并存储，格式为tsv
        srcPath: 读取文本的位置
        outPath: 输出的文本的位置
        label: 标签
        spec: 分割文本和标签的格式
        balence: 是否均衡样本个数

        return: 添加过标签的二维数组
    '''
    res = []
    text = load_text(srcPath)
    que_lines = list(text.iloc[:, 0])

    # 存储路径
    if not os.path.exists(OUTDIR):
        os.makedirs(OUTDIR)

    # 随机挑选句子补齐样本数量
    if blance != 0:
        print('正在补齐样本')
        size = len(que_lines)
        pbar1 = ProgressBar(blance - size)
        for i in range(blance - size):
            rad = random.choice(que_lines)
            que_lines.append(rad)
            pbar1.update(i)

    size = len(que_lines)
    pbar2 = ProgressBar(size)
    fileName = outPath
    f = open(fileName, 'a', encoding='utf-8')

    print('增添标签')
    for i in range(size):
        if byword:
            que_lines[i] = spiltCN_by_word(que_lines[i], stopwords=' ',
                                           isFlag=False)
        else:
            que_lines[i] = splitCN_by_char(que_lines[i], stopwords=' ')
        output = ' '.join(que_lines[i]) + '\t' + label + '\n'
        res.append(output)
        f.write(output)
        pbar2.update(i)
    f.close()

    return res


def clf():
    # 分三类
    src_list = [(config.class_chat, config.LABEL_CHAT),
                (config.class_joke, config.LABEL_JOKE),
                (config.class_QA, config.LABEL_QA)]
    for tup in src_list:
        src = tup[0]
        label = tup[1]
        # 按次分割
        make_fasttext_corpus(src, config.classify_flaged_word,
                             label=label, blance=BALANCE_NUM,
                             byword=True)
        # 按字分割
        make_fasttext_corpus(src, config.classify_flaged_char,
                             label=label, blance=BALANCE_NUM,
                             byword=False)


if __name__ == '__main__':
    clf()

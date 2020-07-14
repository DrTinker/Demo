import os
import pickle
import sys
import threading
sys.path.append(os.getcwd())
import config

from lib import normalize_string, regular
from lib import spiltCN_by_word, splitCN_by_char
from lib import load_stop, load_text
from tqdm import tqdm


def ui():
    mode = input('请输入分词形式(word or char):')
    pseg = False
    if mode == 'word':
        chose = input('是否返回词性(y/n):')
        if chose == 'y':
            pseg = True
        elif chose == 'n':
            pseg = False
        else:
            print('输入无效')
            return None
    elif mode != 'char':
        print('输入无效')
        return None

    return mode, pseg


def del_blank(que_lines, ans_lines):

    que_res = []
    ans_res = []
    length = len(que_lines)

    for i in range(length):
        if len(que_lines[i]) != 0 and len(ans_lines[i]) != 0:
            que_res.append(que_lines[i])
            ans_res.append(ans_lines[i])

    return que_res, ans_res


def split_data(mode, pseg):
    # 分成问答两个数组
    raw_data = load_text(config.xiaohuangji_path, spec='\t', isDF=True)

    que_lines = list(raw_data.iloc[:, 0])
    ans_lines = list(raw_data.iloc[:, 1])

    # 去除多余字符
    print('正在标准化字符')
    que_lines = [normalize_string(regular(data)) for data in que_lines]
    ans_lines = [normalize_string(regular(data)) for data in ans_lines]
    # 标准化后可能出现空行
    que_lines, ans_lines = del_blank(que_lines, ans_lines)
    print('共操作{}条问题文档'.format(len(que_lines)))
    print('共操作{}条回答文档'.format(len(ans_lines)))
    print('标准化字符完毕')

    # 分词
    ans_data, que_data = [], []

    def split(lines, result, type='回答'):
        # 这个语料库以短对话为主，去掉停用词之后导致大量对话长度只有2左右
        # 严重影响训练效果，因此这里不用去掉停用词
        stopwords = ' '
        pbar = tqdm()
        for line in lines:
            # 取出停用词
            if mode == 'word':
                line = spiltCN_by_word(line, stopwords, isFlag=pseg)
            elif mode == 'char':
                line = splitCN_by_char(line, stopwords)
            result.append(line)
            pbar.set_description("正在为{type}分词：".format(type=type))
            pbar.update()
        pbar.close()

    t1 = threading.Thread(target=split, args=(que_lines, que_data, '回答'))
    t2 = threading.Thread(target=split, args=(ans_lines, ans_data, '问题'))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    que_data, ans_data = del_blank(que_data, ans_data)

    return que_data, ans_data


def save_data(filePath, fileName, data, mode='wb'):
    if not os.path.exists(filePath):
        os.makedirs(filePath)

    with open(filePath + fileName, mode=mode) as file:
        pickle.dump(data, file)
    file.close()


def main():
    res = ui()
    if res is None:
        return
    else:
        (mode, pseg) = res
    que_data, ans_data = split_data(mode, pseg)

    filePath = config.xiaohuangji_split_path
    fileNames = ['/splited_que.txt', '/splited_ans.txt']
    print('存储数据')
    save_data(filePath, fileNames[0], que_data)
    save_data(filePath, fileNames[1], ans_data)


def step1(mode, pseg):
    que_data, ans_data = split_data(mode, pseg)

    filePath = config.xiaohuangji_split_path
    fileNames = ['/splited_que.txt', '/splited_ans.txt']
    print('存储数据')
    save_data(filePath, fileNames[0], que_data)
    save_data(filePath, fileNames[1], ans_data)



if __name__ == '__main__':
    main()

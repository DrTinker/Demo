import logging
import os
import sys
sys.path.append(os.getcwd())
import config

import jieba
jieba.setLogLevel(logging.INFO)
jieba.set_dictionary(config.jieba_dict)
jieba.initialize()
import jieba.posseg as pseg


def spiltCN_by_word(sentence, stopwords, isFlag=False):
    '''
        按词语分割一行数据
        sentence: 要分割的一行
        stopwords: 停用词，不需要的话直接输入''
        isFlag: 是否返回词性

        return: isFlag为True，则返回带词性的list其中元素为tuple(词语, 词性)
                e.g: [('计算机', 'n'), ......]
                isFlag为False，则返回不带词性的list，其元素为str
                e.g: ['计算机', ......]
    '''
    res = []
    if not isFlag:
        # 先拆开成单个词
        wordList = jieba.lcut(sentence, cut_all=False)
        for word in wordList:
            if word not in stopwords:
                res.append(word)
        if len(res) == 0:
            return wordList

    else:
        wordList = pseg.cut(sentence)
        for word, flag in wordList:
            if word not in stopwords:
                res.append((word, flag))
        if len(res) == 0:
            for word, flag in wordList:
                res.append((word, flag))

    return res


def splitCN_by_char(sentence, stopwords):
    '''
        按单个字分割一行数据，英文单词不会被拆成字母
        sentence: 要分割的一行
        stopwords: 停用词，不需要的话直接输入''

        return: 返回不带词性的list，其元素为char
                e.g: ['计', '算', '机'......]
    '''
    res = []
    not_split = ''
    jieba.setLogLevel(logging.INFO)

    for char in sentence:
        # 是停用词直接跳过
        if char in stopwords:
            continue
        # 如果是英文，或者数字
        if (char >= u'\u0041' and char <= u'\u005a'):
            not_split += char
            continue
        elif (char >= u'\u0061' and char <= u'\u007a'):
            not_split += char
            continue
        elif char >= u'\u0030' and char <= u'\u0039':
            not_split += char
            continue
        # 如果是中文
        else:
            if not_split != '' or not_split not in stopwords:
                res.append(not_split)
                not_split = ''
            # 这里要把char转成str
            res.append(char + '')

    # 如果只是单独输入了一个英文单词或者数字，就可能出现直接返回空的情况，导致分类器报错
    if len(res) == 0:
        res.append(not_split)

    return res

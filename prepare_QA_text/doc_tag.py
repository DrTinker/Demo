import sys
import os
sys.path.append(os.getcwd())

from lib import spiltCN_by_word, splitCN_by_char
from lib import load_text
from gensim.models.doc2vec import TaggedDocument


def tag_docs(srcPath, byword=False):
    '''
        准备Doc2vec语料
    '''
    res = []
    text = load_text(srcPath)
    que_lines = list(text.iloc[:, 0])

    docs = []
    for doc in que_lines:
        if byword:
            doc = spiltCN_by_word(doc, stopwords=' ', isFlag=False)
        else:
            doc = splitCN_by_char(doc, stopwords=' ')
        docs.append(' '.join(doc))

    for i, doc in enumerate(docs):
        word_list = doc.split(' ')
        size = len(word_list)
        word_list[size-1] = word_list[size-1].strip()
        # print(word_list)
        document = TaggedDocument(word_list, tags=[i])
        res.append(document)

    return res


def get_ques(srcPath, byword=False):
    '''
        获取所有问题，返回值为['我 爱 你'，'你 是谁 ？' .......]
    '''
    text = load_text(srcPath)
    que_lines = list(text.iloc[:, 0])

    docs = []
    for doc in que_lines:
        if byword:
            doc = spiltCN_by_word(doc, stopwords=' ', isFlag=False)
        else:
            doc = splitCN_by_char(doc, stopwords=' ')
        docs.append(' '.join(doc))

    return docs

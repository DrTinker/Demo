import sys
import os
import pickle
import pysparnn.cluster_index as ci
sys.path.append(os.getcwd())
import config

from gensim.models import Doc2Vec
from prepare_QA_text import tag_docs, get_ques
from lib import Bm25Vectorizer

paths = [config.class_chat, config.class_QA, config.class_joke]
names = ['/chat.model', '/QA.model', '/joke.model']


def train_Doc2vec():
    print('训练Doc2vec模型')
    for i, path in enumerate(paths):
        print('正在训练{}'.format(names[i]))
        train = tag_docs(path, byword=True)
        model = Doc2Vec(train, min_count=1, window=3,
                        size=300, sample=1e-3, negative=5, workers=4)

        f = config.doc2vec_save_path + names[i]
        if not os.path.exists(config.doc2vec_save_path):
            os.makedirs(config.doc2vec_save_path)
        model.save(f)


def train_BM():
    print('训练BM模型')
    matrixs = ['./chat.cp', './QA.cp', './joke.cp']
    for i, path in enumerate(paths):
        print('正在训练{}'.format(names[i]))
        train = get_ques(path, byword=True)
        Bm25 = Bm25Vectorizer(norm="l2")
        Bm25.fit(train)
        res = Bm25.transform(train)

        if not os.path.exists(config.BM_save_path):
            os.makedirs(config.BM_save_path)

        tv = config.BM_save_path + names[i]
        with open(tv, 'wb')as f1:
            pickle.dump(Bm25, f1)
        features_list = ci.MultiClusterIndex(res, train)
        cp = config.BM_save_path + matrixs[i]
        with open(cp, 'wb')as f2:
            pickle.dump(features_list, f2)


if __name__ == '__main__':
    train_Doc2vec()
    train_BM()

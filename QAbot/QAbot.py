import os
import sys
import random
import pickle
sys.path.append(os.getcwd())
import config

from lib import load_text, spiltCN_by_word
from lib import normalize_string, regular
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

import warnings
warnings.filterwarnings("ignore")


class QAbot():
    def __init__(self, label):
        super().__init__()
        with open(config.BM_save_path + '/' + label + '.cp', 'rb')as f:
            self.cp = pickle.load(f)
        with open(config.BM_save_path + '/' + label + '.model', 'rb')as f:
            self.tv = pickle.load(f)
        self.dic = load_text(config.class_path + '/' + label + '.tsv',
                             isDF=False, isPrint=False)
        self.QA = load_text(config.class_path + '/' + label + '.tsv',
                            isDF=True, isPrint=False)
        self.model = Doc2Vec.load(config.doc2vec_save_path + '/' + label + '.model')

    def tag(self, docs):
        res = []
        for i, doc in enumerate(docs):
            word_list = doc.split(' ')
            size = len(word_list)
            word_list[size-1] = word_list[size-1].strip()
            # print(word_list)
            document = TaggedDocument(word_list, tags=[i])
            res.append(document)
        return res

    def predict(self, sentence, isBM=False):
        # 先分词
        sentence = spiltCN_by_word(sentence, ' ')
        sentence = [' '.join(sentence)]
        if isBM:
            # 根据BM值先进行初步筛选，获取10个相似度最高的
            search_vec = self.tv.transform(sentence)
            res = self.cp.search(search_vec, k=1, k_clusters=2, return_distance=True)
            # print(res)
            doc = res[0][0][1].replace(' ', '')
            sim = res[0][0][0]
            # pysparnn返回的相似度居然是np.str_，真尼玛奇葩
            sim = float(sim)
            try:
                # 这里的ans是一个list
                ans = self.dic[doc]
            except KeyError:
                return '超出了我的知识范围'
            return (ans, sim)

        else:
            # 为用户输入进行doc2vec
            inferred_vector = self.model.infer_vector(doc_words=sentence,
                                                      alpha=0.025, steps=300)
            # 计算余弦相似度
            sims = self.model.docvecs.most_similar([inferred_vector], topn=3)
            que = []
            ans = []
            for index, sim in sims:
                que.append(self.QA['query'][index])
                ans.append((self.QA['answer'][index], sim))

            # print('最相似的问题:{}，相似度{}'.format(que[0], sims[0][1]))
            # print('chatbot说:{}'.format(ans))
            return ans


def test():
    print('初始化chatbot')
    qabot = QAbot('chat')

    print('初始化完成')
    print('输入exit以结束对话')
    pre = ''
    isqa = False
    while True:
        print('你说：')
        text = sys.stdin.readline()
        if text == 'exit\n':
            print('对话结束')
            return
        text = normalize_string(regular(text))
        if text is None:
            return '电波无法到达哟'

        res_list = qabot.predict(text, isBM=isqa)
        if isqa:
            ans_list = res_list[0]
            random.shuffle(ans_list)
            # print(ans_list)

            while True:
                ans = random.choice(ans_list)
                if pre != ans:
                    pre = ans
                    print('chatbot说：{}'.format(ans))
                    print('sim{}'.format(res_list[1]))
                    break
        else:
            random.shuffle(res_list)
            print(res_list)
            while True:
                (ans, sim) = random.choice(res_list)
                if pre != ans:
                    pre = ans
                    print('chatbot说：{}'.format(ans))
                    break


if __name__ == '__main__':
    test()

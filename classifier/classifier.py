import os
import sys
sys.path.append(os.getcwd())
import config
import fasttext
from lib import spiltCN_by_word, splitCN_by_char


LIMIT = config.predit_limit


class Classifier():
    def __init__(self):
        self.word_model = fasttext.load_model(config.classify_word_model_save)
        self.char_model = fasttext.load_model(config.classify_char_model_save)

    def predict(self, sentence, isPrint=False):
        word = spiltCN_by_word(sentence, '', isFlag=False)
        char = splitCN_by_char(sentence, '')
        # print(char)
        # 预测, 返回结果格式为([['label']], [np.array[score]])
        res_word = self.word_model.predict(word)
        res_char = self.char_model.predict(char)
        # print(res_char)

        p1 = res_word[1][0][0]
        p2 = res_char[1][0][0]
        l1 = res_word[0][0][0]
        l2 = res_char[0][0][0]

        if isPrint:
            print('按词:标签:{}, 概率:{}'.format(l1, p1))
            print('按字:标签:{}, 概率:{}'.format(l2, p2))


        if p1 < LIMIT and p2 < LIMIT:
            return config.LABEL_CHAT
        else:
            if p1 > p2:
                return l1
            else:
                return l2

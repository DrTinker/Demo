import os
import sys
import tensorflow as tf
from gensim.models import Word2Vec
sys.path.append(os.getcwd())
import config

from lib import load_dict, spiltCN_by_word, splitCN_by_char
from lib import normalize_string, regular

from chatbot import instantiate, load_weight, make_embed_init
import math as m

BEAM_SIZE = config.BEAM_SIZE
LAMDA = 0.0001  # 防止出现0


class Chatbot():
    def __init__(self, num):
        super().__init__()
        self.dictionary = load_dict()
        self.stopwords = ' '
        self.model = Word2Vec.load(config.word2vec_model)
        self.matrix = make_embed_init(self.model, self.dictionary)
        self.encoder, self.decoder = instantiate(self.dictionary, self.matrix)

        load_weight(num, self.encoder, self.decoder)

    def get_key(self, dict, value):
        '''
            value转key
        '''
        for k, v in dict.items():
            if v == value:
                return k
        return '<UNK>'

    def update_tensor_value(self, tensor, pos, weight):
        '''
            增加惩罚项，连续生成重复值
        '''
        # 将原来的张量拆分为3部分，修改位置前的部分，要修改的部分和修改位置之后的部分
        i = pos
        part1 = tensor[0][:i]
        part2 = tensor[0][i+1:]
        # 如果是负数，乘以惩罚项反而变大了
        if tensor[0][i].numpy() < 0:
            weight = 2 - weight
        val = tf.constant([tensor[0][i].numpy() * weight], dtype=tf.float32)

        new_tensor = tf.concat([part1, val, part2], axis=0)

        return tf.expand_dims(new_tensor, axis=0)

    # beam search
    def beam_search_decoder(self, data, k):
        '''
            实现beamsearch
        '''
        sequences = [[list(), 0.0]]
        for row in data:
            all_candidates = list()
            for i in range(len(sequences)):
                seq, score = sequences[i]
                for j in range(len(row)):
                    candidate = [seq + [j],
                                 score + -m.log(abs(row[j]) + LAMDA)]
                    all_candidates.append(candidate)
            # 所有候选根据分值排序
            ordered = sorted(all_candidates, key=lambda tup: tup[1])
            # 选择前k个
            sequences = ordered[:k]
        return sequences

    def predict(self, sentence, isBS=False, byword=False):
        '''
            进行预测
            isBM: 是否采用beamsearch
        '''
        # 分词、去除停用词
        if not byword:
            splited = splitCN_by_char(sentence, self.stopwords)
        else:
            splited = spiltCN_by_word(sentence, self.stopwords, isFlag=False)

        # 转化为向量
        vec = []
        for word in splited:
            try:
                vec.append(self.dictionary[word])
            except KeyError:
                vec.append(self.dictionary['<UNK>'])
                continue

        # 将向量输入模型
        en_initial_list = self.encoder.init_state_list(1)
        en_initial_tuple = self.encoder.init_state_tuple(1)
        en_outputs = self.encoder(tf.constant([vec]), en_initial_list, en_initial_tuple)

        de_input = tf.constant([[self.dictionary['<SOS>']]])
        # print('input：{}'.format(de_input))
        de_state_h, de_state_c = en_outputs[1]

        # 使用greedy_search
        if not isBS:
            out_words = []
            # 惩罚因子
            pos = 0
            pre = 0
            punish_weight = config.punish_weight
            punish_weight_max = config.punish_weight_max
            # 循环获取decoder输出，指导<EOS>，或者句子过长
            while True:
                de_output, de_state_h, de_state_c = self.decoder(
                    de_input, (de_state_h, de_state_c), en_outputs[0])
                # 获取预测值 --> 使用gready search转为id --> 通过字典获取词 --> 将词转为200维词向量
                # --> 扩展维度以适应解码器输入
                # 给上一次出现的项乘以惩罚因子，连续出现则惩罚增加
                if pre == pos:
                    de_output = self.update_tensor_value(de_output, pos, punish_weight_max)
                else:
                    de_output = self.update_tensor_value(de_output, pos, punish_weight)

                wid = tf.argmax(de_output, -1).numpy()
                de_input = tf.constant([wid])
                word = self.get_key(self.dictionary, wid[0])
                if word == '<UNK>':
                    word = ''
                # 更新惩罚项
                pre = pos
                pos = wid[0]

                out_words.append(word)
                if out_words[-1] == '<EOS>':
                    out_words = out_words[0:-1]
                    break
                elif len(out_words) >= 20:
                    break

        else:
            # 获取第一次
            de_output, de_state_h, de_state_c = self.decoder(
                    de_input, (de_state_h, de_state_c), en_outputs[0])
            res = self.beam_search_decoder(de_output.numpy(), BEAM_SIZE)
            outputs = []
            out_words = []
            # print('input：{}'.format(res))
            # 循环获取decoder输出，指导<EOS>，或者句子过长
            while True:
                # 保存前一个状态，在下次循环时更新
                pre_de_state_h = de_state_h
                pre_de_state_c = de_state_c

                # 查看当前字符是否为<EOS>或者生成的回答是否过长
                # 这是当前score最大的生成序列中最后一个字符对应的字典序
                cur_id = res[-1][0][-1]
                # print(res)
                # print(out_words)
                if cur_id == self.dictionary['<EOS>'] or len(out_words) >= 20:
                    break
                else:
                    out_words.append(self.get_key(self.dictionary, cur_id))

                # 将每个beam的预测值都带入decoder计算，得到beam_size平方个结果
                # 然后通过score筛选出最大的前beam_size个
                for i in range(BEAM_SIZE):
                    de_input = tf.expand_dims([res[i][0][-1]], axis=0)
                    de_output, de_state_h, de_state_c = self.decoder(
                        de_input, (pre_de_state_h, pre_de_state_c), en_outputs[0])
                    outputs.append(de_output)
                for j in range(len(outputs)-1):
                    de_output = tf.concat([outputs[j], outputs[j+1]], axis=0)
                res = self.beam_search_decoder(de_output.numpy(), BEAM_SIZE)

        print(out_words)
        return ''.join(out_words)


def test():
    print('初始化chatbot')
    chatbot = Chatbot(150)

    print('初始化完成')
    print('输入exit以结束对话')
    while True:
        print('你说：')
        text = sys.stdin.readline()
        if text == '\n':
            print('电波无法到达哟')
            continue
        if text == 'exit\n':
            print('对话结束')
            return
        text = normalize_string(regular(text))
        if text is None:
            print('电波无法到达哟')
            continue

        response = chatbot.predict(text, isBS=False, byword=False)
        print('chatbot说：{}'.format(response))
        # 保存聊天记录
        # my_open = open('./data/chat_record', 'a')
        # my_open.write('human:' + text + '\n')
        # my_open.write('chatbot:' + response + '\n')
        # my_open.close()


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    test()

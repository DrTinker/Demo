import sys
import os
# 使用CPU进行预测
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# 屏蔽GPU加载信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.append(os.getcwd())
import config

from chatbot import Chatbot
from QAbot import QAbot
from classifier import Classifier
from lib import normalize_string, regular
import random


LIMIT = config.SEQ_LIMIT


def keep_record(text, response):
    # 保存聊天记录
    if not os.path.exists(config.chat_record_path):
        os.makedirs(config.chat_record_path)
    my_open = open(config.chat_record, 'a')
    my_open.write('human:' + text + '\n')
    my_open.write('chatbot:' + response + '\n')
    my_open.write('\n')
    my_open.close()


def test():
    print('初始化chatbot')
    clf = Classifier()
    chatbot = Chatbot(100)
    qabot = QAbot('QA')
    chatter = QAbot('chat')
    joker = QAbot('joke')
    label_dict = {config.LABEL_CHAT: 'chat', config.LABEL_GREET: 'greet',
                  config.LABEL_JOKE: 'joke', config.LABEL_QA: 'QA'}
    print('初始化完成')

    print('输入exit以结束对话')
    while True:
        print('你说：')
        text = sys.stdin.readline()
        if text == 'exit\n':
            print('对话结束')
            return
        # 标准化字符
        text = normalize_string(regular(text))

        if len(text) == 0 or text is None:
            print('电波无法到达哟')
            continue

        # 先进行分类
        class_res = clf.predict(text, isPrint=False)
        label = label_dict[class_res]
        print('label{}'.format(label))

        # 实例化对应类别问答机器
        response = ''   # 防止QAbot重复生成回答

        if label != 'chat':
            if label == 'QA':
                res_list = qabot.predict(text, isBM=True)
            elif label == 'joke':
                res_list = joker.predict(text, isBM=True)
            ans_list = res_list[0]
            random.shuffle(ans_list)
            while True:
                ans = random.choice(ans_list)
                if response != ans:
                    response = ans
                    print('chatbot说：{}'.format(response))
                    break
        else:
            res_list = chatter.predict(text, isBM=True)
            ans_list = res_list[0]
            sim = res_list[1]
            # print(sim)
            while True:
                ans = random.choice(ans_list)
                if response != ans:
                    if sim > LIMIT:
                        print('启动生成模式，可能会有些智障')
                        response = chatbot.predict(text, isBS=False, byword=False)
                        print('chatbot说：{}'.format(response))
                        break
                    else:
                        response = ans
                        print('chatbot说：{}'.format(response))
                        break

        keep_record(text, response)


if __name__ == '__main__':
    test()

import os
import sys
sys.path.append(os.getcwd())
import config

import fasttext

EPOCHS = config.FT_EPOCHS


def ft_train():
    print('正在训练分类模型')
    # 存储路径
    if not os.path.exists(config.classify_model_save_path):
        os.makedirs(config.classify_model_save_path)
    # 按词
    word_model = fasttext.train_supervised(config.classify_flaged_word,
                                           wordNgrams=1, epoch=EPOCHS)
    # 按字
    char_model = fasttext.train_supervised(config.classify_flaged_char,
                                           wordNgrams=2, epoch=EPOCHS)
    # 查看训练状况
    print('按词语切分')
    result = word_model.test_label(config.classify_flaged_word)
    for k, v in result.items():
        print(k+':')
        print('precision:{}'.format(v['precision']))
        print('recall:{}'.format(v['recall']))
    print('按字切分')
    result = char_model.test_label(config.classify_flaged_char)
    for k, v in result.items():
        print(k+':')
        print('precision:{}'.format(v['precision']))
        print('recall:{}'.format(v['recall']))

    # 创建文件
    f1 = open(config.classify_word_model_save, 'w')
    f2 = open(config.classify_char_model_save, 'w')
    f1.close()
    f2.close()
    # 保存
    word_model.save_model(config.classify_word_model_save)
    char_model.save_model(config.classify_char_model_save)
    print('分类模型训练结束')


if __name__ == '__main__':
    ft_train()

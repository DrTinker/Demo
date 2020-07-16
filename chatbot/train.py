import sys
import os
sys.path.append(os.getcwd())

from chatbot import Encoder
from chatbot import Decoder
from gensim.models import Word2Vec
from lib import load_dict
from lib import ProgressBar
from lib import shutDown
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pickle
import config


BATCH_SIZE = config.BATCH_SIZE
EMBEDDING_SIZE = config.EMBEDDING_SIZE
RNN_SIZE = config.RNN_UNITS
NUM_EPOCHS = 61
# DICT_SIZE = 50000
DATA_SET_NUM = config.DATA_SET_NUM
TEACHER_FORCE = config.TEACHER_FORCE
CLIPNORM = config.CLIPNORM
LR = config.LR
ATTENTION = config.ATTENTION
METHOD = config.METHOD
DROPOUT = config.DROPOUT
DR = config.DR

ori_dict = config.ori_dict


def instantiate(dictionary, embedding_matrix):
    print('模型实例化')
    # 编码器实例化
    vocab_size = len(dictionary) + 1
    encoder = Encoder(vocab_size, EMBEDDING_SIZE,
                      RNN_SIZE, embedding_matrix,
                      dropout=DROPOUT,
                      dropout_rate=DR)
    # 解码器实例化
    decoder = Decoder(vocab_size, EMBEDDING_SIZE,
                      RNN_SIZE, embedding_matrix,
                      method=METHOD, Bah_atten=ATTENTION,
                      dropout=DROPOUT,
                      dropout_rate=DR)
    initial_tuple = encoder.init_state_tuple(BATCH_SIZE)
    initial_list = encoder.init_state_list(BATCH_SIZE)
    initial_embed = tf.random.uniform([BATCH_SIZE, 1],
                                      minval=0, maxval=100,
                                      dtype=tf.dtypes.int32)
    encoder_outputs = encoder(initial_embed, initial_list, initial_tuple)

    decoder(initial_embed, encoder_outputs[1], encoder_outputs[0])

    return encoder, decoder


# 损失函数
def loss_func(targets, logits):
    crossentropy = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True)
    # 去除填充的PAD和UNK
    no_pad = tf.math.equal(targets, ori_dict['<PAD>'])
    no_unk = tf.math.equal(targets, ori_dict['<UNK>'])
    no_pad_unk = no_pad | no_unk
    mask = tf.math.logical_not(no_pad_unk)
    # bool型转int，用于计算
    mask = tf.cast(mask, dtype=tf.int64)
    # 交叉熵损失
    loss = crossentropy(targets, logits, sample_weight=mask)

    return loss


def acc_func(targets, logits):
    acc = 0
    tag = 0
    for j in range(BATCH_SIZE):
        y = tf.cast(targets[j], dtype=tf.int32)
        if y == ori_dict['<PAD>'] or y == ori_dict['<UNK>']:
            tag += 1
            continue
        if tf.equal(tf.cast(logits[j], dtype=tf.int32), y):
            acc += 1
    return acc, tag


def train_step(source_seq, target_seq_in, target_seq_out,
               en_initial_list, en_initial_tuple,
               encoder, decoder,
               teacher_force=False):
    loss = 0
    acc1 = 0
    tag1 = 0
    with tf.GradientTape() as tape:
        en_outputs = encoder(source_seq, en_initial_list,
                             en_initial_tuple)
        en_states = en_outputs[1]
        (de_state_h, de_state_c) = en_states

        # 不采用teacher_forcing的话需要给定第一个<SOS>
        if not teacher_force:
            decoder_in = tf.constant(value=1, shape=(BATCH_SIZE, 1))
        # 按字拆分开来，使用Teacher Force
        for i in range(target_seq_out.shape[1]):
            if teacher_force:
                # 矩阵维度对齐
                decoder_in = tf.expand_dims(target_seq_in[:, i], 1)
                logit, de_state_h, de_state_c = decoder(
                    decoder_in, (de_state_h, de_state_c), en_outputs[0])
                wid = tf.argmax(logit, -1)
            else:
                logit, de_state_h, de_state_c = decoder(
                    decoder_in, (de_state_h, de_state_c), en_outputs[0])
                # 更新输入，用上一个decoder单元的输出代替输入
                wid = tf.argmax(logit, -1)
                decoder_in = tf.expand_dims(wid, axis=1)

            acc, tag = acc_func(target_seq_out[:, i], wid)
            acc1 += acc
            tag1 += tag
            # print('logit{}'.format(logit.shape))
            # 整个batch学完后loss
            loss += loss_func(target_seq_out[:, i], logit)

    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    # 优化器——梯度下降
    optimizer = tf.keras.optimizers.Adam(learning_rate=LR, clipnorm=CLIPNORM)
    optimizer.apply_gradients(zip(gradients, variables))

    loss = loss / target_seq_out.shape[1]
    acc = acc1 / (target_seq_out.shape[1] * BATCH_SIZE - tag1)

    return loss, acc


def train(encoder, decoder, dataset_list):
    # 训练
    en_initial_tuple = encoder.init_state_tuple(BATCH_SIZE)
    en_initial_list = encoder.init_state_list(BATCH_SIZE)

    for (source_seq, target_seq_in, target_seq_out) in dataset_list:

        source_seq = tf.constant(source_seq)
        target_seq_in = tf.constant(target_seq_in)
        target_seq_out = tf.constant(target_seq_out)
        loss, acc = train_step(source_seq, target_seq_in,
                               target_seq_out,
                               en_initial_list,
                               en_initial_tuple,
                               encoder, decoder,
                               teacher_force=TEACHER_FORCE)

        #if (acc >= 0.95):
        #    print('acc高于0.95，提前终止！')

        #    encoder.save_weights(
        #        config.encoder_model_path + '/encoder_final.h5')
        #    decoder.save_weights(
        #        config.decoder_model_path + '/decoder_final.h5')

        #    exit(0)

        return loss, acc


def make_embed_init(model, dictionary):
    embedding_matrix = np.zeros((len(dictionary) + 1, EMBEDDING_SIZE))
    for word, i in dictionary.items():
        embedding_vector = model.wv[word]
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    return embedding_matrix


# 绘制学习曲线
def draw_graph(data_list, epoch_num=None, mode='loss'):
    # 保存
    if not os.path.exists('./picture/{}'.format(mode)):
        os.makedirs('./picture/{}'.format(mode))
    plt.figure(num='learn_graph', figsize=(6, 3),
               dpi=75, facecolor='#FFFFFF', edgecolor='#FF0000')
    if epoch_num is not None:
        y_list = data_list
        x_list = list(range(0, len(y_list)))
        plt.xticks(list(np.arange(1, NUM_EPOCHS, 10)))
        plt.plot(x_list, y_list, color='red')
        plt.title('{} Graph per Epoch（Epoch{}）'.format(mode, epoch_num))

        plt.savefig('./picture/{}/epoch{}.png'
                    .format(mode, epoch_num))
    else:
        print('未指定epoch num')
        return

    plt.close()


def load_weight(num, encoder, decoder):
    if num < 0:
        return
    # 载入模型
    encoder_checkpoint = config.encoder_model_path
    decoder_checkpoint = config.decoder_model_path
    weights_type = '.h5'
    epoches_num = num

    encoder.load_weights(encoder_checkpoint + '/encoder_'
                         + str(epoches_num) + weights_type)
    decoder.load_weights(decoder_checkpoint + '/decoder_'
                         + str(epoches_num) + weights_type)


def step3():
    print('正在初始化')
    dictionary = load_dict()
    # train(encoder, decoder, teacher_force, dataset_list[0:3])
    model = Word2Vec.load(config.word2vec_model)
    embedding_matrix = make_embed_init(model, dictionary)
    encoder, decoder = instantiate(dictionary, embedding_matrix)
    print('初始化完成')

    print('请输入训练epoch数和起始节点')
    while True:
        print('Epoch_num:')
        NUM_EPOCHS = int(sys.stdin.readline())
        if NUM_EPOCHS <= 0:
            print('训练次数不能为0或者负数')
            continue

        print('Start_node:')
        node = int(sys.stdin.readline())
        if node > 0:
            load_weight(node, encoder, decoder)
        elif node < 0:
            print('请输入有效地起始节点')
            continue

        print('是否使用teacher forceing:(y/n)?')
        temp = sys.stdin.readline()
        if temp == 'y\n':
            TEACHER_FORCE = True
            break
        elif temp == 'n\n':
            TEACHER_FORCE = False
            break
        else:
            print('请重新输入:')
            continue
    print(TEACHER_FORCE)

    # 保存模型
    if not os.path.exists(config.encoder_model_path):
        os.makedirs(config.encoder_model_path)
    if not os.path.exists(config.decoder_model_path):
        os.makedirs(config.decoder_model_path)

    print(encoder.summary())
    print(decoder.summary())

    loss_per_epoch = []
    acc_per_epoch = []
    # 一个EPOCH学所有的dataset
    # 相当于部所有数据分批读入，减小内存压力
    path = config.dataset_path
    DATA_SET_NUM = len([lists for lists in os.listdir(path) if os.path.isfile(os.path.join(path, lists))])
    for e in range(NUM_EPOCHS):
        print('Epoch {}'.format(e+1))

        # tg = ThreadGenerator(t)
        pbar = ProgressBar(DATA_SET_NUM)
        for num in range(DATA_SET_NUM):
            # 读取dataset
            # print('读取dataset{}'.format(num+1))
            with open(config.dataset_path + '/dataset{}'.format(num+1),
                      mode='rb') as file:
                dataset_list = pickle.load(file)
            # print('共读取{}个batch'.format(100))
            loss, acc = train(encoder, decoder, dataset_list)
            pbar.update(num)

        loss_per_epoch.append(loss)
        acc_per_epoch.append(acc)
        print('Epoch{} Loss {:.4f} Acc {:.4f}\n'.format(
            e+1, loss.numpy(), acc))

        if e % 50 == 0 or e == NUM_EPOCHS-1:
            # 保存模型
            encoder.save_weights(
                config.encoder_model_path + '/encoder_{}.h5'.format(e + node))
            decoder.save_weights(
                config.decoder_model_path + '/decoder_{}.h5'.format(e + node))
    encoder.save_weights(
        config.encoder_model_path + '/encoder_final.h5')
    decoder.save_weights(
        config.decoder_model_path + '/decoder_final.h5')
    # 每个epoch中所有数据集的表现
    draw_graph(loss_per_epoch, e+node, mode='loss')
    draw_graph(acc_per_epoch, e+node, mode='acc')


def train_for_ui(epochs, start, isTF=False, isShut=False, isPint=False):
    '''
        通过GUI调用的训练函数
        DATA_SET_NUM要自己根据语料数量来该
        epochs：训练轮数
        start：起始节点，用来加载已经训练好的模型
        isTF：是否使用Tearch Forcing
        isShut：训练结束是否关机
        isPint：是否绘制acc和loss图像
    '''
    print('正在初始化')
    dictionary = load_dict()
    model = Word2Vec.load(config.word2vec_model)
    embedding_matrix = make_embed_init(model, dictionary)
    encoder, decoder = instantiate(dictionary, embedding_matrix)
    print('初始化完成')
    # 判断训练轮数是否为有效值
    if epochs <= 0:
        print('训练次数应当为正整数')
        return
    # 载入模型权重
    if start > 0:
        load_weight(start, encoder, decoder)
    elif start < 0:
        print('请输入有效地起始节点')
        return

    # 保存模型
    if not os.path.exists(config.encoder_model_path):
        os.makedirs(config.encoder_model_path)
    if not os.path.exists(config.decoder_model_path):
        os.makedirs(config.decoder_model_path)

    print(encoder.summary())
    print(decoder.summary())

    loss_per_epoch = []
    acc_per_epoch = []
    # 一个EPOCH学所有的dataset
    # 相当于部所有数据分批读入，减小内存压力
    path = config.dataset_path
    DATA_SET_NUM = len([lists for lists in os.listdir(path) if os.path.isfile(os.path.join(path, lists))])
    for e in range(epochs):
        print('Epoch {}'.format(e+1))

        pbar = ProgressBar(DATA_SET_NUM)
        for num in range(DATA_SET_NUM):
            # 读取dataset
            # print('读取dataset{}'.format(num+1))
            with open(config.dataset_path + '/dataset{}'.format(num+1),
                      mode='rb') as file:
                dataset_list = pickle.load(file)
            # print('共读取{}个batch'.format(100))
            loss, acc = train(encoder, decoder, dataset_list)
            pbar.update(num)

        loss_per_epoch.append(loss)
        acc_per_epoch.append(acc)
        print('Epoch{} Loss {:.4f} Acc {:.4f}\n'.format(
            e+1, loss.numpy(), acc))

        if e % 50 == 0 or e == epochs-1:
            # 保存模型
            encoder.save_weights(
                config.encoder_model_path + '/encoder_{}.h5'.format(e + start))
            decoder.save_weights(
                config.decoder_model_path + '/decoder_{}.h5'.format(e + start))

    # 不管啥情况，最后退出的时候都保存一下
    encoder.save_weights(
        config.encoder_model_path + '/encoder_final.h5')
    decoder.save_weights(
        config.decoder_model_path + '/decoder_final.h5')
    # 每个epoch中所有数据集的表现
    if isPint:
        draw_graph(loss_per_epoch, e+start, mode='loss')
        draw_graph(acc_per_epoch, e+start, mode='acc')
    if isShut:
        shutDown(1)


if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(gpus)
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    step3()
    # 一分钟后关机，不需要可以注释掉
    # shutDown(1)

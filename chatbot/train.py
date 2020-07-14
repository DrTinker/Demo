import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.getcwd())
import config
from lib import shutDown
from lib import ProgressBar
from chatbot import Embedding
from chatbot import Decoder, Projection
from chatbot import Encoder


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


def instantiate():
    print('模型实例化')
    # 词嵌入层实例化
    initial_embed = tf.random.uniform([100, 1],
                                      minval=0, maxval=100,
                                      dtype=tf.dtypes.int32)
    embedding = Embedding(EMBEDDING_SIZE)
    out = embedding(initial_embed)
    # 编码器实例化
    encoder = Encoder(RNN_SIZE,
                      dropout=DROPOUT,
                      dropout_rate=DR)
    # 解码器
    decoder = Decoder(RNN_SIZE,
                      method=METHOD, Bah_atten=ATTENTION,
                      dropout=DROPOUT,
                      dropout_rate=DR)
    # 投影层
    projection = Projection(embedding.tokenLib_size)
    initial_tuple = encoder.init_state_tuple(100)
    initial_list = encoder.init_state_list(100)

    encoder_outputs = encoder(out, initial_list, initial_tuple)

    decoder_outputs = decoder(out, encoder_outputs[1], encoder_outputs[0])
    projection(decoder_outputs[0])

    return embedding, encoder, decoder, projection


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


def train_step(source_seq, target_seq_in, target_seq_out,
               en_initial_list, en_initial_tuple,
               embedding, encoder, decoder, projection,
               teacher_force=False):
    loss = 0
    acc = 0
    with tf.GradientTape() as tape:
        # 先embedding
        en_embed = embedding(source_seq)
        # 编码器
        en_outputs = encoder(en_embed, en_initial_list,
                             en_initial_tuple)
        en_states = en_outputs[1]
        (de_state_h, de_state_c) = en_states

        # 不采用teacher_forcing的话需要给定第一个<SOS>
        if not teacher_force:
            decoder_in = tf.constant(value=1, shape=(BATCH_SIZE, 1))
        # 解码器按字拆分开来
        for i in range(target_seq_out.shape[1]):
            if teacher_force:
                # 矩阵维度对齐
                decoder_in = tf.expand_dims(target_seq_in[:, i], 1)
                de_embed = embedding(decoder_in)
                att_vec, de_state_h, de_state_c = decoder(
                    de_embed, (de_state_h, de_state_c), en_outputs[0])
                # 经过投射层才是输出
                logit = projection(att_vec)
                wid = tf.argmax(logit, -1)
            else:
                de_embed = embedding(decoder_in)
                att_vec, de_state_h, de_state_c = decoder(
                    de_embed, (de_state_h, de_state_c), en_outputs[0])
                # 经过投射层才是输出
                logit = projection(att_vec)
                # 更新输入，用上一个decoder单元的输出代替输入
                wid = tf.argmax(logit, -1)
                decoder_in = tf.expand_dims(wid, axis=1)

            for j in range(BATCH_SIZE):
                if tf.equal(tf.cast(wid[j], dtype=tf.int32),
                            tf.cast(target_seq_out[:, i][j], dtype=tf.int32)):
                    acc += 1
            # print('logit{}'.format(logit.shape))
            # 整个batch学完后loss
            loss += loss_func(target_seq_out[:, i], logit)

    variables = (encoder.trainable_variables + decoder.trainable_variables +
                 embedding.trainable_variables + projection.trainable_variables)
    gradients = tape.gradient(loss, variables)
    # 优化器——梯度下降
    optimizer = tf.keras.optimizers.Adam(learning_rate=LR, clipnorm=CLIPNORM)
    optimizer.apply_gradients(zip(gradients, variables))

    loss = loss / target_seq_out.shape[1]
    acc = acc / (target_seq_out.shape[1] * BATCH_SIZE)

    return loss, acc


def train(embedding, encoder, decoder, projection, dataset_list):
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
                               embedding, encoder,
                               decoder, projection,
                               teacher_force=TEACHER_FORCE)

        if (acc >= 0.8):
            print('acc高于0.8，提前终止！')
            return

        return loss, acc


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
    embedding, encoder, decoder, projection = instantiate()
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

    print(embedding.summary())
    print(encoder.summary())
    print(decoder.summary())
    print(projection.summary())

    loss_per_epoch = []
    acc_per_epoch = []
    # 一个EPOCH学所有的dataset
    # 相当于部所有数据分批读入，减小内存压力
    path = config.dataset_path
    DATA_SET_NUM = len([lists for lists in os.listdir(
        path) if os.path.isfile(os.path.join(path, lists))])
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
            loss, acc = train(embedding, encoder,
                              decoder, projection, dataset_list)
            pbar.update(num)

        loss_per_epoch.append(loss)
        acc_per_epoch.append(acc)
        print('Epoch{} Loss {:.4f} Acc {:.4f}\n'.format(
            e+1, loss.numpy(), acc))

        if e % 50 == 0 or e == NUM_EPOCHS-1:
            # 保存模型
            embedding.save_weights(
                config.model_path + '/embed.h5')
            encoder.save_weights(
                config.encoder_model_path + '/encoder_{}.h5'.format(e + node))
            decoder.save_weights(
                config.decoder_model_path + '/decoder_{}.h5'.format(e + node))
            projection.save_weights(
                config.model_path + '/project.h5')

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
    embedding, encoder, decoder, projection = instantiate()
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

    print(embedding.summary())
    print(encoder.summary())
    print(decoder.summary())
    print(projection.summary())

    loss_per_epoch = []
    acc_per_epoch = []
    # 一个EPOCH学所有的dataset
    # 相当于部所有数据分批读入，减小内存压力
    path = config.dataset_path
    DATA_SET_NUM = len([lists for lists in os.listdir(
        path) if os.path.isfile(os.path.join(path, lists))])
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
            loss, acc = train(embedding, encoder,
                              decoder, projection, dataset_list)
            pbar.update(num)

        loss_per_epoch.append(loss)
        acc_per_epoch.append(acc)
        print('Epoch{} Loss {:.4f} Acc {:.4f}\n'.format(
            e+1, loss.numpy(), acc))

        if e % 50 == 0 or e == epochs-1:
            # 保存模型
            embedding.save_weights(
                config.model_path + '/embed.h5')
            encoder.save_weights(
                config.encoder_model_path + '/encoder_{}.h5'.format(e + start))
            decoder.save_weights(
                config.decoder_model_path + '/decoder_{}.h5'.format(e + start))
            projection.save_weights(
                config.model_path + '/project.h5')

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

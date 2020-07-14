import os
import sys
import numpy as np
import tensorflow as tf
from gensim.models import Word2Vec
sys.path.append(os.getcwd())
import config
from lib import load_dict


class Embedding(tf.keras.Model):
    def __init__(self, embedding_size):
        super(Embedding, self).__init__()
        self.embedding_size = embedding_size
        # 加载模型和字典
        self.word2vec = Word2Vec.load(config.word2vec_model)
        self.dictionary = load_dict()
        embed_init = self.make_embed_init(self.word2vec, self.dictionary)
        self.tokenLib_size = len(self.dictionary) + 1
        # 将向量矩阵用于初始化次嵌入层
        self.embedding = tf.keras.layers.Embedding(
            input_dim=self.tokenLib_size, output_dim=embedding_size,
            embeddings_initializer=tf.keras.initializers.Constant(embed_init))

    def call(self, sequence):
        # 将二维矩阵(batch, maxlen)输入embedding得到(batch, maxlen, embedding_size)
        embed = self.embedding(sequence)
        return embed

    def make_embed_init(self, model, dictionary):
        embedding_matrix = np.zeros((len(dictionary) + 1, self.embedding_size))
        for word, i in dictionary.items():
            embedding_vector = model.wv[word]
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector

        return embedding_matrix

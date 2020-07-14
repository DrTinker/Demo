import os
import sys
import tensorflow as tf
sys.path.append(os.getcwd())


# Attention
class BahdanauAttention(tf.keras.layers.Layer):

    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, decoder_state, encoder_output):
        # decoder_output == (batch_size, rnn_size)
        # hidden_with_time_axis 的形状 ==(batch_size, 1,rnn_size)
        # 这样做是为了执行加法以计算分数
        hidden_with_time_axis = tf.expand_dims(decoder_state, 1)

        # 分数的形状 == (batch_size, max_len, 1)
        # 我们在最后一个轴上得到 1， 因为我们把分数应用于 self.V
        # 在应用 self.V 之前，张量的形状是(batch_size, max_len, rnn_size)
        score = self.V(tf.nn.tanh(
            self.W1(encoder_output) + self.W2(hidden_with_time_axis)))

        # 注意力权重 （attention_weights） 的形状 == (batch_size, max_len, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # 上下文向量 （context_vector） 求和之后的形状 == (batch_size, rnn_size)
        context_vector = attention_weights * encoder_output
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class LuongAttention(tf.keras.layers.Layer):
    def __init__(self, method, units):
        super(LuongAttention, self).__init__()
        self.method = method
        self.rnn_size = units
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "请选择正确的对齐方式")
        if self.method == 'general':
            # htT * Wa * hs
            self.Wa = tf.keras.layers.Dense(self.rnn_size)
        elif self.method == 'concat':
            # VT * tanh(Wa * cat[ht:hs])
            self.Wa = tf.keras.layers.Dense(self.rnn_size,
                                            activation='tanh')
            self.V = tf.keras.layers.Dense(1)

    def call(self, decoder_output, encoder_output):
        # encoder_output的shape=(batch_size, max_len, rnn_size)
        # decoder_output shape=(batch_size, 1, rnn_size)
        if self.method == 'dot':
            score = self.dot_score(decoder_output, encoder_output)
        if self.method == 'general':
            score = self.general_score(decoder_output, encoder_output)
        if self.method == 'concat':
            score = self.concat_score(decoder_output, encoder_output)

        # 计算注意力权重 attention_weights --> (batch_size, 1, rnn_size)
        attention_weights = tf.nn.softmax(score, axis=2)
        # 计算上下文向量
        context_vector = tf.matmul(attention_weights, encoder_output)

        return context_vector, attention_weights

    def dot_score(self, decoder_output, encoder_output):
        score = tf.matmul(decoder_output, encoder_output, transpose_b=True)
        # score(batch_size, 1, max_len)
        return score

    def general_score(self, decoder_output, encoder_output):
        attn_general = self.Wa(encoder_output)
        # attn_general(batch_size, max_len, rnn_size)
        score = tf.matmul(decoder_output, attn_general, transpose_b=True)
        # score(batch_size, 1, max_len)
        return score

    def concat_score(self, decoder_output, encoder_output):
        # decoder_output(batch, 1, rnn_size) --> (batch, maxlen, rnn_size)
        decoder_output = tf.tile(
                decoder_output, [1, encoder_output.shape[1], 1])
        # (batch_size, max_len, 2 * rnn_size) => (batch_size, max_len, rnn_size) => (batch_size, max_len, 1)
        attn_concat = self.V(
            self.Wa(tf.concat([decoder_output, encoder_output], axis=2)))
        # score(batch_size, 1, max_len)
        score = tf.transpose(attn_concat, [0, 2, 1])
        return score

import os
import sys
import tensorflow as tf
sys.path.append(os.getcwd())

from chatbot import BahdanauAttention
from chatbot.attention import LuongAttention



# Decoder
class Decoder(tf.keras.Model):
    '''
        跟Encoder差不多，最后输出的时候多个dense层把RNN的输出映射到词汇表的空间
        rnn_output:(batch_size, max_len, rnn_units) --> final_output(batch_size,
        max_len, tokenLib_size)
    '''
    def __init__(self, tokenLib_size, embedding_size,
                 rnn_units, embed_init,
                 method, Bah_atten=False,
                 dropout=False, dropout_rate=0.0):
        super(Decoder, self).__init__()
        self.rnn_units = rnn_units
        self.Bah_atten = Bah_atten
        self.embedding = tf.keras.layers.Embedding(
            input_dim=tokenLib_size, output_dim=embedding_size,
            embeddings_initializer=tf.keras.initializers.Constant(embed_init))

        self.lstm1 = tf.keras.layers.LSTM(
            units=rnn_units, return_sequences=True, return_state=True)
        self.lstm2 = tf.keras.layers.LSTM(
            units=rnn_units, return_sequences=True, return_state=True)
        self.lstm3 = tf.keras.layers.LSTM(
            units=rnn_units, return_sequences=True, return_state=True)
        self.dropout = dropout
        if dropout:
            self.drop1 = tf.keras.layers.Dropout(rate=dropout_rate)
            self.drop2 = tf.keras.layers.Dropout(rate=dropout_rate)

        self.dense = tf.keras.layers.Dense(tokenLib_size)
        self.wc = tf.keras.layers.Dense(rnn_units, activation='tanh')

        # 用于attention
        if self.Bah_atten:
            self.attention = BahdanauAttention(self.rnn_units)
        else:
            self.attention = LuongAttention(method, self.rnn_units)

    def call(self, sequence, states_truple, encoder_output):
        if self.Bah_atten:
            # 算出attention weight和context vector
            context_vector, attention_weights = self.attention(states_truple[0],
                                                               encoder_output)

            # 先经过Embedding层形成稠密矩阵
            embed = self.embedding(sequence)

            # 将生成的稠密矩阵输入LSTM，同时将初始状态h、c也输入LSTM
            lstm_out, _, _ = self.lstm1(embed, initial_state=states_truple)
            if self.dropout:
                lstm_out = self.drop1(lstm_out)
            lstm_out, _, _ = self.lstm2(lstm_out, initial_state=states_truple)
            if self.dropout:
                lstm_out = self.drop2(lstm_out)
            lstm_out, state_h, state_c = self.lstm3(lstm_out,
                                                    initial_state=states_truple)

            # 计算attention_vector
            lstm_out = tf.concat(
                [context_vector, tf.squeeze(lstm_out, axis=1)], axis=1)
            attention_vector = self.wc(lstm_out)

        else:
            # 先经过Embedding层形成稠密矩阵
            embed = self.embedding(sequence)

            # 将生成的稠密矩阵输入LSTM，同时将初始状态h、c也输入LSTM
            lstm_out, _, _ = self.lstm1(embed, initial_state=states_truple)
            if self.dropout:
                lstm_out = self.drop1(lstm_out)
            lstm_out, _, _ = self.lstm2(lstm_out, initial_state=states_truple)
            if self.dropout:
                lstm_out = self.drop2(lstm_out)
            lstm_out, state_h, state_c = self.lstm3(lstm_out,
                                                    initial_state=states_truple)
            # 算出attention weight和context vector
            context_vector, attention_weights = self.attention(lstm_out,
                                                               encoder_output)

            # 计算attention_vector
            lstm_out = tf.concat(
                [tf.squeeze(context_vector, axis=1),
                 tf.squeeze(lstm_out, axis=1)], axis=1)
            attention_vector = self.wc(lstm_out)

        # Dense层进行映射
        output = self.dense(attention_vector)

        # log_softmax归一化
        output = tf.nn.log_softmax(output)

        return output, state_h, state_c

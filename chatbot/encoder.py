import tensorflow as tf


# Encoder 编码器和没有注意力机制的一样
# embedding --> lstm
class Encoder(tf.keras.Model):
    '''
        encoder是一个双层双向的LSTM，并且embedding层采用Word2vec训练好的模型初始化权重
        tokenLib_size:表示词汇表大小
        embedding_size:表示经过Embedding层后生成的稠密向量的第三个维度数
        rnn_units:表示一个LSTM Cell中每个门的前馈神经网络中神经元的个数（这里参见CSDN的博客解析
        embd_init:是word2vec训练好的向量矩阵，维度是(batch_size, max_len)即(100, 200)

        在Encoder中，矩阵维数变化如下：
        (batch_size, max_len) --> Embedding --> (batch_size, max_len,
        embedding_size) --> LSTM --> 输出三个
        output:(batch_size, max_len, 2*rnn_units) hidden_state:(batch_size,
        rnn_units)
        cell_state:(batch_size, rnn_units)
        这里双向LSTM会将fw和bw的输出结果按最后一个维度，即rnn_units拼接
        然后再经过一个单向lstm，最终输出向量维度(batch_size, max_len, rnn_units)
    '''
    def __init__(self, tokenLib_size, embedding_size, rnn_units,
                 embed_init, dropout=False, dropout_rate=0.0):
        super(Encoder, self).__init__()
        self.rnn_units = rnn_units

        self.embedding = tf.keras.layers.Embedding(
            input_dim=tokenLib_size, output_dim=embedding_size,
            embeddings_initializer=tf.keras.initializers.Constant(embed_init))

        # 第一层
        self.fw_lstm1 = tf.keras.layers.LSTM(
            units=rnn_units, return_sequences=True,
            return_state=True)
        self.bw_lstm1 = tf.keras.layers.LSTM(
            units=rnn_units, return_sequences=True,
            return_state=True, go_backwards=True)
        # 第二层
        self.fw_lstm2 = tf.keras.layers.LSTM(
            units=rnn_units, return_sequences=True,
            return_state=True)
        self.bw_lstm2 = tf.keras.layers.LSTM(
            units=rnn_units, return_sequences=True,
            return_state=True, go_backwards=True)
        # 第三层
        self.last_lstm = tf.keras.layers.LSTM(
            units=rnn_units, return_sequences=True,
            return_state=True)
        if dropout:
            self.drop1 = tf.keras.layers.Dropout(rate=dropout_rate)
            self.drop2 = tf.keras.layers.Dropout(rate=dropout_rate)
            self.dropout = dropout


    def call(self, sequence, states_list, state_truple):
        # 先经过Embedding层形成稠密矩阵
        embed = self.embedding(sequence)
        # 前向和后向LSTM
        fw_out, _, _ = self.fw_lstm1(embed, initial_state=states_list[0])
        bw_out, _, _ = self.bw_lstm1(embed, initial_state=states_list[1])
        output1 = tf.concat([fw_out, bw_out], axis=-1)
        if self.dropout:
            output1 = self.drop1(output1)
        # 第二层
        fw_out, _, _ = self.fw_lstm2(output1, initial_state=states_list[0])
        bw_out, _, _ = self.bw_lstm2(output1, initial_state=states_list[1])
        # 按照rnn_units维度拼接output
        output1 = tf.concat([fw_out, bw_out], axis=-1)
        if self.dropout:
            output1 = self.drop2(output1)

        output2, last_h, last_c = self.last_lstm(output1, state_truple)

        state_truple = (last_h, last_c)
        return output2, state_truple

    # 定义两个全0维度为(batch_size, rnn_size)的矩阵，作为初始的hidden state和cell state
    def init_state_list(self, batch_size):
        return [(tf.zeros([batch_size, self.rnn_units]), tf.zeros([batch_size,
                self.rnn_units])),
                (tf.zeros([batch_size, self.rnn_units]), tf.zeros([batch_size,
                 self.rnn_units]))]

    def init_state_tuple(self, batch_size):
        return (tf.zeros([batch_size, self.rnn_units]),
                tf.zeros([batch_size, self.rnn_units]))

# 停止词路径
stopword_path = './corpus/stopword/stopword.txt'

# 小黄鸡语料库路径
xiaohuangji_path = './corpus/xiaohuangji.tsv'

# 小黄鸡分词文本存储路径
xiaohuangji_split_path = './corpus/data/splited'
xiaohuangji_split_que = './corpus/data/splited/splited_que.txt'
xiaohuangji_split_ans = './corpus/data/splited/splited_ans.txt'
xiaohuangji_pad_path = './corpus/data/pad'
xiaohuangji_pad_que = './corpus/data/pad/padding_que.txt'
xiaohuangji_pad_ans = './corpus/data/pad/padding_ans.txt'

# 提前准备好的类别
class_path = './corpus/classify'
class_chat = './corpus/classify/chat.tsv'
class_greet = './corpus/classify/greet.tsv'
class_joke = './corpus/classify/joke.tsv'
class_QA = './corpus/classify/QA.tsv'
class_self = './corpus/classify/self.tsv'

# 输出增添标签后的类别文本
classify_flaged_path = './corpus/data/classify'
classify_flaged_word = './corpus/data/classify/fasttext_by_word'
classify_flaged_char = './corpus/data/classify/fasttext_by_char'

# 准备分类文本时统一样本数量
BALANCE_NUM = 500

# 分类标签
LABEL_CHAT = '__label__chat'
LABEL_JOKE = '__label__joke'
LABEL_GREET = '__label__greet'
LABEL_QA = '__label__QA'

# fasttext训练的超参数
FT_EPOCHS = 100

# fasttext模型保存路径
classify_model_save_path = './model/classify'
classify_word_model_save = './model/classify/classify_word.model'
classify_char_model_save = './model/classify/classify_char.model'

# 分类器阈值
predit_limit = 0.5

# Embedding层的word2vec训练超参数
MAX_FREQUENCE = 50000    # 最高出现频率
MIN_FREQUENCE = 5   # 最低出现频率
EMBEDDING_SIZE = 200    # 词向量维度
KEEP_WORD = ['你好', '我', '他', '你']   #经过筛选后强制保留的词
LEN_OF_QUE = 8     # padding后的同一句子长度
LEN_OF_ANS = 12
WINDOW = 4  #cbow或者skip-gram模型窗口大小
MODE = 'skip-gram'  # 选择的模型cbow或者skip-gram

# 初始字典
ori_dict = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}

# word2vec存储路径
word2vec_path = './model/word2vec'
word2vec_model = './model/word2vec/word2vec.model'

# 预测时的惩罚因子
punish_weight = 0.8
punish_weight_max = 0.4
BEAM_SIZE = 3

# 想在已有模型上进行增量训练
APPEND = False

# 字典存储路径
dict_path = './corpus/data/dictionary'
dict_file = './corpus/data/dictionary/dictionary.txt'
jieba_dict = './corpus/data/dictionary/dict.txt'
jieba_user_dict = './corpus/data/dictionary/user_dict.txt'

# 训练seq2seq的超参数
BATCH_SIZE = 100

# dataset存储路径
dataset_path = './corpus/data/dataset'
DATA_SET_SIZE = 100

# seq2seq的超参数
ATTENTION = False   # 是否使用Bah Atten
METHOD = 'general'  # 使用Luong Atten时的score方法
RNN_UNITS = 512
DATA_SET_NUM = 46
TEACHER_FORCE = False
CLIPNORM = 1e-07
LR = 0.001
DROPOUT = True
DR = 0.3

# checkpoint保存路径
model_path = './model/checkpoints_luong'
encoder_model_path = './model/checkpoints_luong/encoder'
decoder_model_path = './model/checkpoints_luong/decoder'
embedding_model = './model/checkpoints_luong/embed.h5'
project_model = './model/checkpoints_luong/project.h5'

# 新增聊天记录保存位置
chat_record_path = './corpus/data/record'
chat_record = './corpus/data/record/record.txt'

# doc2vec模型保存路径
doc2vec_save_path = './model/doc2vec'
doc2vec_save_chat = './model/doc2vec/chat.model'
doc2vec_save_QA = './model/doc2vec/QA.model'
doc2vec_save_joke = './model/doc2vec/joke.model'

# bm模型保存路径
BM_save_path = './model/BM'
BM_cp_QA = './model/BM/QA.cp'
BM_cp_chat = './model/BM/chat.cp'
BM_cp_joke = './model/BM/joke.cp'
BM_TV_QA = './model/BM/QA.model'
BM_TV_chat = './model/BM/chat.model'
BM_TV_joke = './model/BM/joke.model'


# 启动生成模式的阈值
SEQ_LIMIT = 0.005

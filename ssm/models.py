import pandas
import jieba
import gensim
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import pandas as pd
import tensorflow as tf
from keras.models import Model, load_model, Sequential
import crfSubprocess

BATCH_SIZE = 256
PATH = './data'
PATH_MODEL = './model'
PATH_RAW_DATA = PATH + '/项目1.txt'
PATH_DATA = PATH +  '/data.csv'
PATH_DATA_DATA0_TRAIN = PATH + '/data0_train.csv'
PATH_DATA_DATA0_TEST = PATH + '/data0_test.csv'
PATH_STOP_WORDS = PATH + '/stopwords.txt'
PATH_WORD2VECT_MODEL = PATH_MODEL + '/word2vec'
PATH_WORD2VECT_DATA = PATH + '/word2vec.csv'
PATH_WORD2VECT_EXTERN_MODEL = PATH_MODEL + '/word2vec_extern_60d'
CLASS_TAG = {'ILF': 0, 'EIF': 1, 'EI' : 2, 'EO': 3, 'EQ': 4}
CLASS_MAP = ['ILF', 'EIF', 'EI', 'EO', 'EQ']
FILTER_WORDS = ['\t', '1', '2', '3', '4']
PATH_TOP_MODEL = PATH_MODEL + '/top_model'
PATH_LEFT_MODEL = PATH_MODEL + '/left_model'
PATH_RIGHT_MODEL = PATH_MODEL + '/right_model'
PATH_MERGED_MODEL = PATH_MODEL + '/merged_model'
MAX_SEQUENCE_LENGTH = 50
WORD2VECT_SIZE = 60

def load_wordvec_model(model_file=PATH_WORD2VECT_MODEL):
    return gensim.models.Word2Vec.load(model_file)  

def load_stop_words(stop_file):
    stopwords = [line.strip() for line in open(stop_file, 'rb').readlines()]  
    return stopwords    

def load_wordvec(csv_file=PATH_WORD2VECT_DATA):
    df = pandas.read_csv(csv_file)
    print(df.shape)
    return df

    
stopwords = load_stop_words(PATH_STOP_WORDS)
wv= load_wordvec()
wv_type = 'dict'
print('loading model')
# top_model = load_model(PATH_TOP_MODEL)
# left_model = load_model(PATH_LEFT_MODEL)
# right_model = load_model(PATH_RIGHT_MODEL)
model = load_model(PATH_MERGED_MODEL)
graph = tf.get_default_graph()
print('load done')

# cut sentence then return a list
def cut_sentence(sentence, filter_words=FILTER_WORDS):
    words = jieba.cut(sentence)
    return [word for word in words if (word not in stopwords and word not in filter_words)]


def predict_preprocess(words_list):
    x = []
    for words in words_list:
        lst = []
        for w in words:
            if w in wv:
                lst.append(wv[w])
            else:
                if wv_type == 'model':
                    lst.append(np.zeros((WORD2VECT_SIZE, )))
                else:
                    lst.append([0] * WORD2VECT_SIZE)
        x.append(lst)
    x = list(pad_sequences(x, maxlen=MAX_SEQUENCE_LENGTH, dtype="float32"))
    return np.array(x)

    
def predict_class_with_mult_model(x):
    l_map = [0, 3]
    r_map = [1, 2, 4]
    # predict
    pred = []
    global graph
    with graph.as_default():
        top_pred = np.argmax(top_model.predict(x, batch_size=BATCH_SIZE), axis=1)
        lp_pred = np.argmax(left_model.predict(x, batch_size=BATCH_SIZE), axis=1)
        rp_pred = np.argmax(right_model.predict(x, batch_size=BATCH_SIZE), axis=1)

    # print('log: predict', x)

    for i in range(0, top_pred.shape[0]):
        if top_pred[i] == 0:
            tag = l_map[int(lp_pred[i])]
            pred.append(CLASS_MAP[tag])
        else:
            tag = r_map[int(rp_pred[i])]
            pred.append(CLASS_MAP[tag])
    
    return pred


def predict_class_with_merged_model(x):
    # 返回预测类别的列表
    global graph
    with graph.as_default():
        print(x)
        num_pred = np.argmax(model.predict(x, batch_size=BATCH_SIZE), axis=1)
    pred = [CLASS_MAP[num] for num in num_pred]
    
    return pred


def predict_class(x):
    return predict_class_with_merged_model(x)


if __name__ == '__main__':
    inp = '制定调查问卷信息，包括投票主题、时间、调研题目等信息'
    tag = 'EIF'
    sentence = []
    model.summary()
    sentence.append(cut_sentence(inp))
    print(sentence)
    x = predict_preprocess(sentence)
    pred = predict_class(x)
    print(pred)
    count_items = crfSubprocess.crf_predict(sentence, pred)
    print(sentence)
    print(count_items)
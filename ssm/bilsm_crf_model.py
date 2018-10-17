from typing import List

from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM
from keras_contrib.layers import CRF
import tensorflow as tf
import pickle
import numpy as np

EMBED_DIM = 200
BiRNN_UNITS = 200
MODEL_PATH = "model_v2/"


def create_model():
    with open('model_v2/config.pkl', 'rb') as inp:
        (vocab, chunk_tags) = pickle.load(inp)
    model = Sequential()
    model.add(Embedding(len(vocab), EMBED_DIM, mask_zero=True))  # Random embedding
    model.add(Bidirectional(LSTM(BiRNN_UNITS // 2, return_sequences=True)))
    crf = CRF(len(chunk_tags), sparse_target=True)
    model.add(crf)
    model.summary()
    model.compile('adam', loss=crf.loss_function, metrics=[crf.accuracy])
    return model, (vocab, chunk_tags)


model, (vocab, chunk_tags) = create_model()
model.load_weights(MODEL_PATH + 'crf.h5')
graph = tf.get_default_graph()

print("Swithin: finish loading model")
word2idx = dict((w, i) for i, w in enumerate(vocab))


def process_sentence(data, maxlen=100):
    x = [word2idx.get(w[0].lower(), 1) for w in data]
    length = len(x)
    # x = pad_sequences([x], maxlen)  # left padding
    # x = [x]
    x = np.array([x])
    return x, length


def predict_sentences(sentences):
    tags = []  # type: List[str]
    count_items = []  # type: List[str]
    global graph
    with graph.as_default():
        for predict_text in sentences:
            senten_vec, length = process_sentence(predict_text)
            model.load_weights(MODEL_PATH + 'crf.h5')
            raw = model.predict(senten_vec)[0]  # [-length:]
            result = [np.argmax(row) for row in raw]
            # chunk_tags = ['O', 'B-EI', 'I-EI', 'B-EO', 'I-EO', 'B-EQ', 'I-EQ', 'B-ILF', 'I-ILF', 'B-EIF', 'I-EIF']
            # result_tags = [chunk_tags[i] for i in result]
            # EI, EO, EQ, ILF, EIF = '', '', '', '', ''
            _count_items = {'EI': '', 'EO': '', 'EQ': '', 'ILF': '', 'EIF': '', '-': '-'}
            flag = '-'
            for s, t in zip(predict_text, result):
                if t == 0:
                    continue
                if t in {1, 2}:  # ('B-EI', 'I-EI')
                    flag = 'EI'
                elif t in {3, 4}:  # ('B-EO', 'I-EO')
                    flag = 'EO'
                elif t in {5, 6}:  # ('B-EQ', 'I-EQ')
                    flag = 'EQ'
                elif t in {7, 8}:  # ('B-ILF', 'I-ILF')
                    flag = 'ILF'
                elif t in {9, 10}:  # ('B-EIF', 'I-EIF')
                    flag = 'EIF'
                if t in {1, 3, 5, 7, 9}:
                    _count_items[flag] += ' '
                _count_items[flag] += s
            print("result_tags:", flag)
            print(_count_items)
            tags.append(flag)
            count_items.append(_count_items[flag])
    return tags, count_items


if __name__ == '__main__':
    inp = []
    y = []
    with open('../nlp_v2.0_sijin/实验/summary_2.txt', 'r', encoding='utf-8', errors='ignore') as inf:
        for line in inf.readlines():
            line = line.strip()
            print("line:", line)
            if line:
                t_vec = line.split('\t')
                if len(t_vec) >= 3:
                    y.append(t_vec[2])
                    dec = t_vec[0].strip()
                    inp.append(dec)
                    count_name = t_vec[1]
    pred = predict_sentences(inp)

    print("same_count", np.sum(np.array(y) == np.array(pred)))
    print("sum_num", len(y))

import numpy as np
from collections import Counter
from keras.preprocessing.sequence import pad_sequences
import keras
import pickle
import platform


def load_data(create_vocab=True, vocab_dir='model/config.pkl'):
    train = _parse_data(open('data/train_data_e.data', 'rb'))
    test = _parse_data(open('data/test_data_e.data', 'rb'))
    # print("train", train)
    if create_vocab:
        # 从数据集中生成新vocab列表
        word_counts = Counter(row[0].lower() for sample in train for row in sample)
        vocab = [w for w, f in iter(word_counts.items()) if f >= 2]
        chunk_tags = ['O', 'B-EI', 'I-EI', 'B-EO', 'I-EO', 'B-EQ', 'I-EQ', 'B-ILF', 'I-ILF', 'B-EIF', 'I-EIF']

        # save initial config data
        with open(vocab_dir, 'wb') as outp:
            pickle.dump((vocab, chunk_tags), outp)
    else:
        with open(vocab_dir, 'rb') as inp:
            (vocab, chunk_tags) = pickle.load(inp)
    train = _process_data(train, vocab, chunk_tags)
    test = _process_data(test, vocab, chunk_tags)
    return train, test, (vocab, chunk_tags)


def _parse_data(fh):
    #  in windows the new line is '\r\n\r\n' the space is '\r\n' . so if you use windows system,
    #  you have to use recorsponding instructions

    if platform.system() == 'Windows':
        split_text = '\n'
    else:
        split_text = '\n'

    string = fh.read().decode('utf-8')
    #data = [[row.split() for row in sample.split(split_text)] for
            #sample in
            #string.strip().split(split_text + split_text)]
    data = []
    for sample in string.strip().split(']'):
        data_tmp = []
        if sample != '':
            for row in sample.split(split_text):
                if row.split() != [] and len(row.split()) == 2:
                    data_tmp.append(row.split())
            data.append(data_tmp)
    fh.close()
    return data


def _process_data(data, vocab, chunk_tags, maxlen=None, onehot=False):
    if maxlen is None:
        maxlen = max(len(s) for s in data)
    word2idx = dict((w, i+1) for i, w in enumerate(vocab))  # 以前从0开始，修正为从1开始
    x = [[word2idx.get(w[0].lower(), 1) for w in s] for s in data]  # set to <unk> (index 1) if not in vocab

    y_chunk = [[chunk_tags.index(w[1]) for w in s] for s in data]

    x = pad_sequences(x, maxlen)  # left padding

    y_chunk = pad_sequences(y_chunk, maxlen, value=-1)

    if onehot:
        y_chunk = np.eye(len(chunk_tags), dtype='float32')[y_chunk]
    else:
        y_chunk = np.expand_dims(y_chunk, 2)
    return x, y_chunk


def process_data(data, vocab, maxlen=100):
    word2idx = dict((w, i+1) for i, w in enumerate(vocab))  # 以前从0开始，修正为从1开始
    x = [word2idx.get(w[0].lower(), 1) for w in data]
    length = len(x)
    # x = pad_sequences([x], maxlen)  # left padding
    # x = [x]
    x = np.array([x])

    return x, length


def y2one_hot(x, y_padded):
    y_max = np.amax(y_padded, axis=1)
    sample_filter = np.where(y_max>0)

    x2 = x[sample_filter, :]
    y2 = y_max[sample_filter]
    y2[(y2%2)==1] += 1
    y2 = y2/2 - 1
    y2 = keras.utils.to_categorical(y2, num_classes=None)
    return x2, y2


if __name__ == '__main__':
    # train = _parse_data(open('data/train_data_e.data', 'rb'))
    # test = _parse_data(open('data/test_data_e.data', 'rb'))
    # # print ("train", train)
    # print(len(train))
    # print(len(test))
    # print(train[0])
    # word_counts = Counter(row[0].lower() for sample in train for row in sample)
    # vocab1 = set([w for w, f in iter(word_counts.items()) if f >= 2])
    # word_counts = Counter(row[0].lower() for sample in test for row in sample)
    # vocab2 = set([w for w, f in iter(word_counts.items()) if f >= 2])
    # vocab3 = vocab1&vocab2
    # print(len(vocab1),len(vocab2),len(vocab3))

    (train_x, train_y), (test_x, test_y), (vocab, chunk_tags) = load_data(
        create_vocab=False, vocab_dir='model/config_w2v_tc.pkl')
    train_x, train_y = y2one_hot(train_x, train_y)
    test_x, test_y = y2one_hot(test_x, test_y)
    print(len(train_y))
    print(len(train_x))
    print(train_y[0:5])
    print(train_x.shape)
    print(test_x.shape)
    print(train_x[0:5])

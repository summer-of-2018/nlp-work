import numpy
from collections import Counter
from keras.preprocessing.sequence import pad_sequences
import pickle
import platform


def load_data():
    # train = _parse_data(open('data/train_data_e.data', 'rb'))
    train = _parse_data(open('data/data_b.data', 'rb'))
    test = _parse_data(open('data/test_data_e.data', 'rb'))
    print("train", train)
    word_counts = Counter(row[0].lower() for sample in train for row in sample)
    vocab = [w for w, f in iter(word_counts.items()) if f >= 2]  # 筛选出频率不止一次的字
    # TODO 能保证多次运行vocab的顺序是一样的吗
    chunk_tags = ['O', 'B-EI', 'I-EI', 'B-EO', 'I-EO', 'B-EQ', 'I-EQ', 'B-ILF', 'I-ILF', 'B-EIF', 'I-EIF']

    # save initial config data
    with open('model/config.pkl', 'wb') as outp:
        pickle.dump((vocab, chunk_tags), outp)

    train = _process_data(train, vocab, chunk_tags,maxlen=218)
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
    # data = [[row.split() for row in sample.split(split_text)] for
    # sample in
    # string.strip().split(split_text + split_text)]
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
        print("maxlen=",maxlen)
    word2idx = dict((w, i) for i, w in enumerate(vocab))
    x = [[word2idx.get(w[0].lower(), 1) for w in s] for s in data]  # set to <unk> (index 1) if not in vocab
    # TODO 为什么word2idx查询不到时用1？
    y_chunk = [[chunk_tags.index(w[1]) for w in s] for s in data]

    x = pad_sequences(x, maxlen)  # left padding

    y_chunk = pad_sequences(y_chunk, maxlen, value=-1)

    if onehot:
        y_chunk = numpy.eye(len(chunk_tags), dtype='float32')[y_chunk]
    else:
        y_chunk = numpy.expand_dims(y_chunk, 2)
    return x, y_chunk


def process_data(data, vocab, maxlen=100):
    word2idx = dict((w, i) for i, w in enumerate(vocab))
    x = [word2idx.get(w[0].lower(), 1) for w in data]
    length = len(x)
    x = pad_sequences([x], maxlen)  # left padding
    return x, length

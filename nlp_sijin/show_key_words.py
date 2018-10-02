import numpy
from collections import Counter
from keras.preprocessing.sequence import pad_sequences
import pickle
import platform


def load_data():
    train = _parse_data(open('data/data_b.data', 'rb'))
    test = _parse_data(open('data/test_data_e.data', 'rb'))
    print("train", train)
    counter_dict = {'O':0, 'B-EI':0, 'I-EI':0, 'B-EO':0, 'I-EO':0, 'B-EQ':0, 'I-EQ':0, 'B-ILF':0, 'I-ILF':0, 'B-EIF':0, 'I-EIF':0}
    key_words = {'EI': set(), 'EO': set(), 'EQ': set(), 'ILF': set(), 'EIF': set()}
    for sample in train:
        key_word = ""
        key_word_label = ""
        for row in sample:
            w = row[0]
            label = row[1]
            counter_dict[label] += 1
            if label in ['B-EI', 'B-EO', 'B-EQ', 'B-ILF', 'B-EIF']:
                if key_word != "":
                    print(key_word_label, key_word)
                    key_words[key_word_label].add(key_word)
                key_word = w
                key_word_label = label.split('-')[1]
            elif label != 'O':
                key_word += w
                key_word_label = label.split('-')[1]
        if key_word != "":
            print(key_word_label, key_word)
            key_words[key_word_label].add(key_word)

    return key_words, counter_dict


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


if __name__ == '__main__':
    key_words, counter_dict = load_data()

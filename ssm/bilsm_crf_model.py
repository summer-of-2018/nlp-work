from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM
from keras_contrib.layers import CRF
import process_data
import pickle
import numpy as np

EMBED_DIM = 200
BiRNN_UNITS = 200


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
word2idx = dict((w, i) for i, w in enumerate(vocab))
count_num = 0
sum_num = 0


def process_sentence(data, maxlen=100):
    x = [word2idx.get(w[0].lower(), 1) for w in data]
    length = len(x)
    # x = pad_sequences([x], maxlen)  # left padding
    # x = [x]
    x = np.array([x])
    return x, length


def predict_sentence(predict_text:str):
    str, length = process_sentence(predict_text)
    model.load_weights('model/crf.h5')
    raw = model.predict(str)[0]  # [-length:]
    result = [np.argmax(row) for row in raw]
    result_tags = [chunk_tags[i] for i in result]
    EI, EO, EQ, ILF, EIF = '', '', '', '', ''
    flag = ''
    for s, t in zip(predict_text, result_tags):
        if t in ('B-EI', 'I-EI'):
            EI += ' ' + s if (t == 'B-EI') else s
            flag = 'EI'
        if t in ('B-EO', 'I-EO'):
            EO += ' ' + s if (t == 'B-EO') else s
            flag = 'EO'
        if t in ('B-EQ', 'I-EQ'):
            EQ += ' ' + s if (t == 'B-EQ') else s
            flag = 'EQ'
        if t in ('B-ILF', 'I-ILF'):
            ILF += ' ' + s if (t == 'B-ILF') else s
            flag = 'ILF'
        if t in ('B-EIF', 'I-EIF'):
            EIF += ' ' + s if (t == 'B-EIF') else s
            flag = 'EIF'
    print("result_tags:", flag)
    if flag == label:
        count_num += 1
    if flag == 'EI':
        EI_count += 1
    elif flag == 'EO':
        EO_count += 1
    elif flag == 'EQ':
        EQ_count += 1
    elif flag == 'ILF':
        ILF_count += 1
    else:
        EIF_count += 1
    if label == 'EI':
        EI_count_f += 1
    elif label == 'EO':
        EO_count_f += 1
    elif label == 'EQ':
        EQ_count_f += 1
    elif label == 'ILF':
        ILF_count_f += 1
    else:
        EIF_count_f += 1
    print(['EI:' + EI, 'EO:' + EO, 'EQ:' + EQ, 'ILF:' + ILF, 'EIF:' + EIF])
    print("label:", label)
    outf.write(line)
    outf.write("\n")
    result_str = 'EI:' + EI + ',' + 'EO:' + EO + ',' + 'EQ:' + EQ + ',' + 'ILF:' + ILF + ',' + 'EIF:' + EIF
    outf.write(result_str)
    outf.write("\n")
    outf.write("\n")

outf = open('实验/test_result.txt', 'w', encoding='utf-8', errors='ignore')
with open('实验/summary_2.txt', 'r', encoding='utf-8', errors='ignore') as inf:
    for line in inf.readlines():
        line = line.strip()
        print("line:", line)
        if line:
            t_vec = line.split('\t')
            if len(t_vec) >= 3:
                sum_num += 1
                dec = t_vec[0]
                dec = dec.strip()
                count_name = t_vec[1]
                label = t_vec[2]
                predict_text = dec
                str, length = process_data.process_data(predict_text, vocab)
                model.load_weights('model/crf.h5')
                raw = model.predict(str)[0][-length:]
                result = [np.argmax(row) for row in raw]
                result_tags = [chunk_tags[i] for i in result]
                EI, EO, EQ, ILF, EIF = '', '', '', '', ''
                flag = ''
                for s, t in zip(predict_text, result_tags):
                    if t in ('B-EI', 'I-EI'):
                        EI += ' ' + s if (t == 'B-EI') else s
                        flag = 'EI'
                    if t in ('B-EO', 'I-EO'):
                        EO += ' ' + s if (t == 'B-EO') else s
                        flag = 'EO'
                    if t in ('B-EQ', 'I-EQ'):
                        EQ += ' ' + s if (t == 'B-EQ') else s
                        flag = 'EQ'
                    if t in ('B-ILF', 'I-ILF'):
                        ILF += ' ' + s if (t == 'B-ILF') else s
                        flag = 'ILF'
                    if t in ('B-EIF', 'I-EIF'):
                        EIF += ' ' + s if (t == 'B-EIF') else s
                        flag = 'EIF'
                print("result_tags:", flag)
                if flag == label:
                    count_num += 1
                if flag == 'EI':
                    EI_count += 1
                elif flag == 'EO':
                    EO_count += 1
                elif flag == 'EQ':
                    EQ_count += 1
                elif flag == 'ILF':
                    ILF_count += 1
                else:
                    EIF_count += 1
                if label == 'EI':
                    EI_count_f += 1
                elif label == 'EO':
                    EO_count_f += 1
                elif label == 'EQ':
                    EQ_count_f += 1
                elif label == 'ILF':
                    ILF_count_f += 1
                else:
                    EIF_count_f += 1
                print(['EI:' + EI, 'EO:' + EO, 'EQ:' + EQ, 'ILF:' + ILF, 'EIF:' + EIF])
                print("label:", label)
                outf.write(line)
                outf.write("\n")
                result_str = 'EI:' + EI + ',' + 'EO:' + EO + ',' + 'EQ:' + EQ + ',' + 'ILF:' + ILF + ',' + 'EIF:' + EIF
                outf.write(result_str)
                outf.write("\n")
                outf.write("\n")
    outf.close()
print("count_num", count_num)
print("sum_num", sum_num)
print("EI_count", EI_count)
print("EO_count", EO_count)
print("EQ_count", EQ_count)
print("ILF_count", ILF_count)
print("EIF_count", EIF_count)
print("EI_count_f", EI_count_f)
print("EO_count_f", EO_count_f)
print("EQ_count_f", EQ_count_f)
print("ILF_count_f", ILF_count_f)
print("EIF_count_f", EIF_count_f)

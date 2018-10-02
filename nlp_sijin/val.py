import bilsm_crf_model
import process_data
import numpy as np

model, (vocab, chunk_tags) = bilsm_crf_model.create_model(train=False)
model.load_weights('model/crf_b.h5')
count_num = 0
sum_num = 0
with open('data/data_b.txt', 'r', encoding='utf-8', errors='ignore') as inf:
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
                str, length = process_data.process_data(predict_text, vocab, maxlen=218)
                raw = model.predict(str)[0][-length:]
                result = [np.argmax(row) for row in raw]
                result_tags = [chunk_tags[i] for i in result]
                EI, EO, EQ, ILF, EIF = '', '', '', '', ''
                flag = ""
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
                if flag == label:
                    count_num += 1
                print(['EI:' + EI, 'EO:' + EO, 'EQ:' + EQ, 'ILF:' + ILF, 'EIF:' + EIF])
                print("result_tags:", flag)
                print("label:", label)
print("count_num", count_num)
print("sum_num", sum_num)

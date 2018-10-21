import bilsm_crf_model
import process_data
import numpy as np

model, (vocab, chunk_tags) = bilsm_crf_model.create_model(train=False)

model.load_weights('model/crf.h5')
s = np.array([[1,4,12,56,32]])
raw1 = model.predict(s)
print(raw1)


model.load_weights('model/crf.h5')
s = np.array([[0,0,0,0,0,0,0,1,4,12,56,32]])
raw2 = model.predict(s)
print(raw2)

model.load_weights('model/crf.h5')
s = np.array([[0,0,0,0,0,0,0,1,4,12,56,32]])
raw3 = model.predict(s)[0][-5:]

count_num = 0
sum_num = 0
EI_count = 0
EO_count = 0
EQ_count = 0
EIF_count = 0
ILF_count = 0
EI_count_f = 0
EO_count_f = 0
EQ_count_f = 0
EIF_count_f = 0
ILF_count_f = 0

row_idx = 1

outf = open('实验/test_result.txt', 'w', encoding='utf-8', errors='ignore')
with open('实验/summary_2.txt', 'r', encoding='utf-8', errors='ignore') as inf:
    for line in inf.readlines():
        line = line.strip()
        print (row_idx, "line:", line)
        row_idx += 1
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

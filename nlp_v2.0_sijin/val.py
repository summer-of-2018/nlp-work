import bilsm_crf_model
import process_data
import numpy as np

model, (vocab, chunk_tags) = bilsm_crf_model.create_model(train=False)
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
outf = open('实验/test_result.txt', 'w', encoding='utf-8', errors='ignore')
with open('实验/summary_2.txt', 'r', encoding='utf-8', errors='ignore') as inf:
	for line in inf.readlines():
		line = line.strip()
		print ("line:", line)
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
print ("count_num", count_num)
print ("sum_num", sum_num)
print ("EI_count", EI_count)
print ("EO_count", EO_count)
print ("EQ_count", EQ_count)
print ("ILF_count", ILF_count)
print ("EIF_count", EIF_count)
print ("EI_count_f", EI_count_f)
print ("EO_count_f", EO_count_f)
print ("EQ_count_f", EQ_count_f)
print ("ILF_count_f", ILF_count_f)
print ("EIF_count_f", EIF_count_f)
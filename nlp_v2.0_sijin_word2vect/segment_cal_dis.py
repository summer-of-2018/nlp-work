# -*- coding=utf-8 -*-
from gensim.models import word2vec
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import preprocessing
import numpy as np
import pandas as pd
import time
import jieba
import re
import types
import tensorflow as tf
from gensim.models import word2vec
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedKFold
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout
from keras.layers.core import Lambda
from keras.layers.merge import concatenate, add, multiply
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import OneHotEncoder
import warnings
import io
import sys
import urllib.request
from keras.preprocessing.text import Tokenizer
sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8',errors='ignore')
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')


# jieba.load_userdict('userdict.txt')  
# 创建停用词list  
def stopwordslist(stop_file):  
    stopwords = [line.strip() for line in open(stop_file, 'r').readlines()]  
    return stopwords  

def cal_word_edit(lhs, rhs):
    #if type(lhs) != types.UnicodeType:
    #    lhs = lhs.decode('utf8')
    #if type(rhs) != types.UnicodeType:
    #    rhs = rhs.decode('utf8')
    len_1 = len(lhs)
    len_2 = len(rhs)
    #dist_table = [[0] * (len_2 + 1) ] * (len_1 + 1)
    dist_table = [[0] * (len_2 + 1) for i in range(len_1 + 1)]
    for i in range(len_1 + 1):
        dist_table[i][0] = i
    for j in range(len_2 + 1):
        dist_table[0][j] = j
    for i in range(1, len_1 + 1):
        for j in range(1, len_2 + 1):
            if lhs[i - 1] == rhs[j - 1]:
                cost = 0
            else:
                cost = 1
            deletion = dist_table[i - 1][j] + 1;
            insertion = dist_table[i][j - 1] + 1;
            substitution = dist_table[i - 1][j - 1] + cost;
            dist_table[i][j] = min(min(deletion, insertion), substitution);
    return dist_table[len_1][len_2];



# 对句子进行分词;赋予标签
def seg_sentence(input_file, log_file, stop_file, segwithoutlabel_file):
	words = []
	labels = []
	text1 = []
	outstr=''
	stopwords = stopwordslist(stop_file)  # 这里加载停用词的路径  
	outf = open(log_file, 'w', encoding='utf-8', errors='ignore')
	outft = open(segwithoutlabel_file, 'w', encoding='utf-8', errors='ignore')
	with open(input_file,'r', encoding='utf-8', errors='ignore') as inf:
		for line in inf.readlines():
			line = line.strip()
			print ("line:",line)
			if line:
				t_vec = line.split('\t')
				if len(t_vec) >= 3:
					dec = t_vec[0]
					dec = dec.strip()
					count_name = t_vec[1]
					label = str(t_vec[2])
					if label == 'ILf':
						label = 'ILF'
					if label == 'EIf':
						label = 'EIF'
				else:
					continue
			labels.append(label)
			print ("dec", dec)
			print ("count_name", count_name)
			print ("label", label)
			sentence_seged = jieba.cut(dec)
			count_seged = jieba.cut(count_name)
			sentence_result = []
			count_result = []
			for word in sentence_seged:
				sentence_result.append(word)
			for count_word in count_seged:
				count_result.append(count_word)
			text2 = []
			print ("sentence_result", sentence_result)
			print ("count_result", count_result)
			for word in sentence_result:
				flag = 'O'
				text1 = []
				if word not in stopwords:
					#print ("stopwords")
					if word != '\t' and word != '' and word != ' ':
						#print ("kongbai")
						for count_e in count_result:
							#print ("daozhelile")
							if count_e == '':
								#print ("count_word", count_e)
								continue
							else:
								if word == count_e:
									flag = label
									break
								else:
									print ("word", word)
									print ("count_word", count_e)
									print ("length", len(count_e))
									dist = cal_word_edit(word, count_e)
									min_len = min(len(word), len(count_e))
									if min_len <= 3:
										if dist < 2:
											flag = label
											break
									elif min_len <= 7:
										if dist < 3:
											flag = label
											break
									else:
										if dist < 4:
											flag = label
											break
						i = 0
						for str_tmp in word:
							if flag == 'O':
								str_flag = 'O'
							elif i == 0:
								str_flag = 'B' + '-' + flag
							else:
								str_flag = 'I' + '-' + flag
							i += 1
							text1.append(str_tmp)
							text1.append(str_flag)
							text2.append(str_tmp)
							outstr = str_tmp.strip()
							outstr = outstr.strip(' ')
							outf.write(outstr)
							outft.write(outstr)
							outf.write(" ")
							outft.write(" ")
							outf.write(str_flag)
							outf.write("\n")
			outf.write("]")
			outf.write("\n")
						#print ("line:",outstr)
	outf.close()
	outft.close()
	return text2



#对分词后的文件进行词向量编辑
def word_embedding(segwithoutlabel_file, log_file, write_file):
	labels=[]
	data=[]
	data_tmp = []
	data_label = []
	sentences = word2vec.Text8Corpus(segwithoutlabel_file)	 # 90% train
	model = word2vec.Word2Vec(sentences, sg=1, size=300, window=5, min_count=1, negative=3,sample=0.001, hs=1, workers=4)
	model.save('word2vec_model')  # save



	outf = open(write_file, 'w', encoding='utf-8', errors='ignore')
	with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
		for line in f.readlines():
			line = line.strip()
			print ("line", line)
			if line:
				t_vec = line.split('\t')
				if len(t_vec) >= 2:
					keyword = t_vec[0]
					label = t_vec[1]
					keyword = keyword.strip()
					try:
						if keyword:
							print ("word:",keyword)
							t_mode = model[keyword]
							data_tmp.append(t_mode)
							labels.append(label)
					except KeyError:
						print ("not in vocabulary:",KeyError)
				else:
					data.append(data_tmp)
					data_label.append(labels)
					data_tmp = []
					labels = []
		datastr = str(data)
		outf.write(datastr)
		outf.write("\n")
		outf.close()
	return data, data_label


#对分词进行编号和pad处理
#得到每个句子的词向量10*30，15*30
# 接下来，把每个句子pad成相同的长度 如20*30
def premodel(data, label):
	MAX_SEQUENCE_LENGTH = 30
	#print ("data:",data)
	tokenizer = Tokenizer(filters="")
	tokenizer.fit_on_texts(np.append(data))
	word_index = tokenizer.word_index
	#print ("word_index",word_index)


	data_1 = pad_sequences(tokenizer.texts_to_sequences(data), maxlen=MAX_SEQUENCE_LENGTH)
	labels = np.array(label)
	ohe = OneHotEncoder()
	ohe.fit([[0],[1],[2],[3],[4],[5]])
	ohlabel = ohe.transform(labels.reshape(-1,1)).toarray()
	print ("ohlabel:",ohlabel)
	return data_1, ohlabel

def model_build(data, ohlabels, test_data, test_labels):

	input_data = Input(shape = (50, 300))
	bilstm = Bidirectional(LSTM(32, dropout_W=0.1, dropout_U=0.1, return_sequences=True))(input_data)
	bilstm_d = Dropout(0.1)(bilstm)
	half_window_size = 2
	paddinglayer = ZeroPadding1D(padding=half_window_size)(input_data)
	conv = Conv1D(nb_filter=50, filter_length=(2 * half_window_size + 1), border_mode='valid')(paddinglayer)
	conv_d = Dropout(0.1)(conv)
	dense_conv = TimeDistributed(Dense(50))(conv_d)
	rnn_cnn_merge = merge([bilstm_d, dense_conv], mode='concat', concat_axis=2)
	class_label_count = 6
	dense = TimeDistributed(Dense(class_label_count))(rnn_cnn_merge)
	crf = CRF(class_label_count, sparse_target=False)
	crf_output = crf(dense)
	model = Model(input=[word_input], output=[crf_output])
	model.summary()


	precision = as_keras_metric(tf.metrics.precision)
	recall = as_keras_metric(tf.metrics.recall)
	model.compile(loss=crf.loss_function, optimizer="adam", metrics=[crf.accuracy, precision, recall])
	early_stopping = EarlyStopping(monitor="val_loss", patience=2)
	best_model_path = "best_model" + str(model_count) + ".h5"
	model_checkpoint = ModelCheckpoint(best_model_path, save_best_only=True, save_weights_only=True)
	try:
		model.load_weights(best_model_path,by_name = True, skip_mismatch = True)
	except:
		pass
	hist = model.fit(data, ohlabels,validation_data=(test_data, test_labels),epochs=5, batch_size=BATCH_SIZE, shuffle=True,callbacks=[early_stopping, model_checkpoint], verbose=2)
		#model.load_weights(best_model_path,by_name = True, skip_mismatch = True)
	try:
		model.load_weights(best_model_path,by_name = True, skip_mismatch = True)
	except:
		pass
	preds_1 = model.predict(test_data, batch_size=BATCH_SIZE, verbose=1)
	return preds_1




#对原始文件进行分词和词向量编码
#train_text=seg_sentence('./lstm/train.txt', './lstm/train分词文件.txt', './stopwords.txt', './lstm/train词向量文件.txt')

#test_text=seg_sentence('./lstm/test.txt', './lstm/test分词文件.txt', './stopwords.txt', './lstm/test词向量文件.txt')

#train_text为词加标签的二维list
train_text=seg_sentence('./实验/summary_2.txt', './实验/test_data_e.data', './实验/stopwords.txt', './实验/segwithoutlabel_file.txt')

#data为词向量集合，二维list，array，label为label的集合
#train_data, train_labels = word_embedding('./segwithoutlabel_file.txt', './train分词文件.txt', './train向量文件.txt')
#train_data = word_embedding('./lstm/train词向量文件.txt', './lstm/train向量文件.txt')
#test_data = word_embedding('./lstm/test词向量文件.txt', './lstm/test向量文件.txt')

#train_data_1, train_labels_1 = premodel(train_data, train_labels)
#test_data_1 = premodel(test_data, test_labels)

#preds_1 = model_build(train_data_1,train_labels,test_data_1,test_labels)

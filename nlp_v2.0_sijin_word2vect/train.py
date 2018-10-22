import bilsm_crf_model
import numpy as np
import os
import pickle

EPOCHS = 10
WORD2VEC_PATH = '../'
EMBEDDING_DIM = 300

model, (train_x, train_y), (test_x, test_y) = bilsm_crf_model.create_model()

# load embedding weights
embeddings_index = dict()  # 所有的词向量
with open(os.path.join(WORD2VEC_PATH, 'merge_sgns_bigram_char300.txt'), encoding='utf-8') as f:
    i = 1
    for line in f:
        if i % 10000 == 1:
            print(i)
        i+=1
        # try:
        values = line.split()
        if len(values) != 301:
            print(i, '!=301')
            print(line)
            continue
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
        # except Exception as e:
        #     print(e)
        #     print(i)
        #     print(line)
print('Found %s word vectors.' % len(embeddings_index))


with open('model/config.pkl', 'rb') as inp:
    (vocab, chunk_tags) = pickle.load(inp)
word2idx = dict((w, i+1) for i, w in enumerate(vocab))


embedding_matrix = np.zeros((len(word2idx) + 1, EMBEDDING_DIM))

for word, i in word2idx.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
del embeddings_index

model.layers[0].trainable = False
model.layers[0].set_weights([embedding_matrix])
crf = model.layers[-1]
model.compile('adam', loss=crf.loss_function, metrics=[crf.accuracy])
model.summary()

# train model
model.fit(train_x, train_y,batch_size=16,epochs=EPOCHS, validation_data=[test_x, test_y])
model.save('model/crf_w2v.h5')

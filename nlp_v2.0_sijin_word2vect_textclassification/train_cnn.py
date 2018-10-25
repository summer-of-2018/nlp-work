import cnn
import numpy as np
import os
import pickle
import process_data
import keras

EPOCHS = 10
WORD2VEC_PATH = '../'
EMBEDDING_DIM = 300
maxlen=250

(train_x, train_y), (test_x, test_y), (vocab, chunk_tags) = process_data.load_sentences(
    create_vocab=True, vocab_dir='model/config_w2v_tc.pkl', maxlen=maxlen)
# train_x, train_y = process_data.y2one_hot(train_x, train_y)
# test_x, test_y = process_data.y2one_hot(test_x, test_y)

# print(train_x[:5])
# print(train_y[:5])
# print(test_x[:5])
# print(test_y[:5])
model = cnn.create_model(maxlen, len(vocab))

# load embedding weights
embeddings_index = dict()  # 所有的词向量
# https://github.com/Embedding/Chinese-Word-Vectors
with open(os.path.join(WORD2VEC_PATH, 'merge_sgns_bigram_char300.txt'), encoding='utf-8') as f:
    i = 1
    for line in f:
        if i % 10000 == 1:
            print(i)
        i+=1
        # try:
        values = line.split()
        if len(values) != 301:
            print(i, '!=301')  # split维数有错
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


# with open('model/config.pkl', 'rb') as inp:
#     (vocab, chunk_tags) = pickle.load(inp)
word2idx = dict((w, i+1) for i, w in enumerate(vocab))


embedding_matrix = np.zeros((len(word2idx) + 1, EMBEDDING_DIM))

for word, i in word2idx.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
del embeddings_index

model.layers[1].trainable = False
model.layers[1].set_weights([embedding_matrix])
model.compile('adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# train model
model.fit(train_x, train_y,batch_size=32,epochs=EPOCHS, validation_data=[test_x, test_y],
          callbacks=[keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, embeddings_freq=0)])


model.layers[1].trainable = True
model.compile('adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(train_x, train_y,batch_size=32,epochs=EPOCHS, validation_data=[test_x, test_y],
          callbacks=[keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, embeddings_freq=0)])



model.save('model/crf_w2v_tc.h5')

import bilsm_crf_model
import numpy as np
import os
import pickle
import keras
import process_data

EPOCHS = 10
WORD2VEC_PATH = '../'
EMBEDDING_DIM = 300

model, (vocab, chunk_tags) = bilsm_crf_model.create_model(train=False)
(train_x, train_y), (test_x, test_y), (vocab, chunk_tags) = process_data.load_data(
    create_vocab=False, vocab_dir='model/config_w2v.pkl')



# model.layers[0].trainable = False
# model.layers[0].set_weights([embedding_matrix])
# crf = model.layers[-1]
# model.compile('adam', loss=crf.loss_function, metrics=[crf.accuracy])
# model.summary()

model.load_weights('model/crf_w2v.h5')

# train model
model.fit(train_x, train_y,batch_size=16,epochs=EPOCHS, validation_data=[test_x, test_y],
          callbacks=[keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1, embeddings_freq=1)])
model.save('model/crf_w2v_finetune.h5')

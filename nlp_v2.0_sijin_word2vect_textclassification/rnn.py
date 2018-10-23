from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM, Dense
from keras_contrib.layers import CRF
import process_data
import pickle

EMBED_DIM = 300  # 200
BiRNN_UNITS_1 = 200
BiRNN_UNITS_2 = 100
num_classes = 5

def create_model(vocab_size):
    model = Sequential()
    model.add(Embedding(vocab_size + 1, EMBED_DIM, mask_zero=True))  # 修改了
    model.add(Bidirectional(LSTM(BiRNN_UNITS_1 // 2, return_sequences=True)))
    model.add(Bidirectional(LSTM(BiRNN_UNITS_2 // 2, return_sequences=False)))
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()
    model.compile('adam', loss='categorical_crossentropy',)
    return model

from keras.models import Sequential, Model
from keras.layers import Embedding, Bidirectional, LSTM, Dense, Input, Conv1D, GlobalMaxPooling1D, concatenate, Dropout
from keras_contrib.layers import CRF
import process_data
import pickle

EMBED_DIM = 300  # 200
BiRNN_UNITS_1 = 200
BiRNN_UNITS_2 = 200
num_classes = 5

num_filters=64
filter_sizes=[3, 4, 5]
dropout_rate=0.5


# def create_model(vocab_size):
#     model = Sequential()
#     model.add(Embedding(vocab_size + 1, EMBED_DIM, mask_zero=True))  # 修改了
#     model.add(Bidirectional(LSTM(BiRNN_UNITS_1 // 2, return_sequences=True)))
#     model.add(Bidirectional(LSTM(BiRNN_UNITS_2 // 2, return_sequences=False)))
#     model.add(Dense(100, activation='relu'))
#     model.add(Dense(num_classes, activation='softmax'))
#     model.summary()
#     model.compile('adam', loss='categorical_crossentropy', metrics=['accuracy'])
#     return model


def create_model(max_tokens, vocab_size):
    sequence_input = Input(shape=(max_tokens,), dtype='int32')
    embedding_layer = Embedding(vocab_size + 1,
                                EMBED_DIM,
                                # weights=[build_embedding_weights(self.token_index, self.embeddings_index)],
                                # input_length=max_tokens,
                                mask_zero=False,
                                # trainable=trainable_embeddings,
                                )
    x = embedding_layer(sequence_input)

    pooled_tensors = []
    for filter_size in filter_sizes:
        x_i = Conv1D(num_filters, filter_size, activation='elu')(x)
        x_i = GlobalMaxPooling1D()(x_i)
        pooled_tensors.append(x_i)

    x = pooled_tensors[0] if len(filter_sizes) == 1 else concatenate(pooled_tensors, axis=-1)

    if dropout_rate > 0:
        x = Dropout(dropout_rate)(x)
    x = Dense(num_classes, activation='softmax')(x)
    model = Model(sequence_input, x)
    model.summary()
    model.compile('adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
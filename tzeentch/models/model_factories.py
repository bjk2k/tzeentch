import tensorflow.python as tf
import os
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.layers import MultiHeadAttention
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.metrics import kl_divergence
from tensorflow.python.keras.layers import CuDNNLSTM
from keras_self_attention import SeqSelfAttention


def make_rnn_model(input_dim, output_dim):
    L1 = 512

    model = Sequential()

    model.add(CuDNNLSTM(L1, input_shape=input_dim, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(output_dim))
    model.add(Activation('softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['acc'])

    model.summary()
    return model


def make_attention_model(input_dim, output_dim, rem_state=False, batch_size=10):
    L1 = 256

    if rem_state:
        input_layer = Input(shape=input_dim, name='Input', batch_size=batch_size)
    else:
        input_layer = Input(shape=input_dim, name='Input')

    att_layer = MultiHeadAttention(num_heads=8, key_dim=8, name='Multi-Head')(input_layer, input_layer)

    lstm = LSTM(L1, input_shape=input_dim, return_sequences=True, stateful=rem_state)(att_layer)
    #lstm = CuDNNLSTM(L1, input_shape=input_dim, return_sequences=True)(att_layer)

    lstm = BatchNormalization()(lstm)
    lstm = Dropout(0.2)(lstm)
    lstm = Flatten()(lstm)
    output = Dense(output_dim, activation='softmax')(lstm)

    model = Model(inputs=input_layer, outputs=output)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc', 'categorical_crossentropy'])

    model.summary()

    return model


def make_autoencoder_model(input_dim, output_dim):
    L1 = 30
    L2 = 30
    L3 = 20
    L4 = 30
    L5 = 40

    print(input_dim)

    input_seq = Input(shape=input_dim)

    encoded = Dense(L1, activation='relu', activity_regularizer=regularizers.l2(0))(input_seq)
    encoded = Dense(L2, activation='relu', activity_regularizer=regularizers.l2(0))(encoded)
    encoded = Dense(L3, activation='relu', activity_regularizer=regularizers.l2(0))(encoded)

    decoded = Dense(L4, activation='relu', activity_regularizer=regularizers.l2(0))(encoded)
    decoded = Dense(L5, activation='relu', activity_regularizer=regularizers.l2(0))(decoded)
    decoded = Dense(output_dim, activation='linear')(decoded)

    autoencoder = Model(input_seq, decoded)
    autoencoder.compile(optimizer='rmsprop', loss='mse', metrics=['acc'])
    autoencoder.summary()

    encoder = Model(input_seq, encoded)

    return encoder, autoencoder

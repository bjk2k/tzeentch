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


def make_autoencoder_model(input_dim, output_dim, hp):
    L1 = hp.Int('layer 1', 15, 50, step=1)
    L2 = hp.Int('layer 2', 20, 60, step=1)
    L3 = hp.Int('layer 3', 25, 70, step=1)
    L4 = hp.Int('layer 4', 30, 80, step=1)
    L5 = hp.Int('layer 5', 40, 90, step=1)

    print(input_dim)

    input_seq = Input(shape=input_dim)

    encoded = Dense(L1, activation='relu', activity_regularizer=regularizers.l2(0))(input_seq)
    encoded = Dense(L2, activation='relu', activity_regularizer=regularizers.l2(0))(encoded)
    encoded = Dense(L3, activation='relu', activity_regularizer=regularizers.l2(0))(encoded)

    decoded = Dense(L4, activation='relu', activity_regularizer=regularizers.l2(0))(encoded)
    decoded = Dense(L5, activation='relu', activity_regularizer=regularizers.l2(0))(decoded)
    decoded = Dense(output_dim, activation='linear')(decoded)

    autoencoder = Model(input_seq, decoded)
    autoencoder.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])
    autoencoder.summary()

    encoder = Model(input_seq, encoded)

    return encoder, autoencoder


def make_attention_model(input_dim, output_dim, hp):
    L1 = hp.Int('lstm_units', 200, 300, step=4)
    NUM_HEADS = hp.Int('num_heads', 1, 10, step=1)
    KEY_DIM = hp.Int('key_dim', 1, 10, step=1)
    DROPOUT = hp.Float('dropout', min_value=0.05, max_value=0.9, step=0.05)
    USE_CUDNN = hp.Boolean('use_cudnn')

    input_layer = Input(shape=input_dim, name='Input')

    att_layer = MultiHeadAttention(num_heads=NUM_HEADS, key_dim=KEY_DIM, name='Multi-Head')(input_layer, input_layer)

    if USE_CUDNN:
        lstm = CuDNNLSTM(L1, input_shape=input_dim, return_sequences=True)(att_layer)
    else:
        lstm = LSTM(L1, input_shape=input_dim, return_sequences=True)(att_layer)

    lstm = BatchNormalization()(lstm)
    lstm = Dropout(DROPOUT)(lstm)
    lstm = Flatten()(lstm)
    output = Dense(output_dim, activation='softmax')(lstm)

    model = Model(inputs=input_layer, outputs=output)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc', 'categorical_crossentropy'])

    model.summary()

    return model

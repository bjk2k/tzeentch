import os.path

import pandas as pd
# %matplotlib inline
# Load the TensorBoard notebook extension
# %load_ext tensorboard

# imports a module for calculating with dates
import datetime as dt
import matplotlib.pyplot as plt

import tensorflow.python.keras.layers

# PARAMETERs

START = dt.datetime.fromisocalendar(2005, 1, 1).date()  # has to be the 1st of January of some year
END = dt.datetime.fromisocalendar(2021, 1, 1).date()

# FILES
path_to_best_encoder = os.path.join("models", "autoencoder-lstm", "val_acc-optimized_enc")
path_to_best_rnn = os.path.join("models", "autoencoder-lstm", "val_acc-optimized_rnn")
path_to_best_att = os.path.join("models", "autoencoder-lstm", "val_acc-optimized_att")

# this imports support for hinting types in methods (only interesting for software)

#
#   Information Retrieval
#
from typing import cast

import tzeentch.stockwrappers

from tzeentch.stockwrappers import DataSource
from tzeentch.stockwrappers import IndexInfo

# reloads every module at restart of the notebook (technical stuff unimportant)
import importlib

importlib.reload(tzeentch.stockwrappers)

# read index price data from imported data source in a given time frame also give the ticker included in StockInfo
# another name for convenience
info_SP500: IndexInfo = cast(IndexInfo, DataSource.retrieve_yfinance('^GSPC', start=START, end=END))

#
# Preprocessing - reindexing and classifying dataset
#

# period model shall predict into future
FUTURE_PERIOD_PREDICT = 14

# input and target columns for the autoencoder model
input_columns_autoenc = ['open', 'high', 'low', 'close', 'volume', 'trend_cci', 'momentum_stoch',
                         'trend_ema20',
                         'adjclose', 'trend_sma_fast', 'trend_sma_slow',
                         'trend_macd', 'volatility_bbh', 'volatility_bbm', 'volatility_bbl', 'momentum_rsi',
                         'volume_cmf'
                         ]
target_columns = ['close']

# renmae and classify
from tzeentch.preprocessing.noise_filters import extract_and_preapre_features
input_df, input_df_only_renamed = extract_and_preapre_features(seq_len=FUTURE_PERIOD_PREDICT,
                                                      stock_info=info_SP500,
                                                      feature_colums=input_columns_autoenc,
                                                      target_columns=target_columns)

# apply scaling
from tzeentch.preprocessing.noise_filters import apply_min_max_scaling

panel_df_train, panel_df_test = apply_min_max_scaling(input_df, target_columns)
panel_df_test_full = input_df_only_renamed[~input_df_only_renamed.index.isin(panel_df_train.index)]

# de-noise using wavelet transforms
from tzeentch.transformer.sequence_transformers import sequence_generator

train_X, train_Y = sequence_generator(panel_df_train, FUTURE_PERIOD_PREDICT, shuffle=True, seed=101)
test_X, test_Y = sequence_generator(panel_df_test, FUTURE_PERIOD_PREDICT, shuffle=False)

#
#   Model   -   Autoencode
#

BATCH_SIZE = 10
N_ITER = 50

from tensorflow.keras.callbacks import ModelCheckpoint
from tzeentch.models.model_factories import make_autoencoder_model

encoder, autoencoder = make_autoencoder_model(train_X.shape[1:], train_X.shape[2])

checkpoint = ModelCheckpoint(path_to_best_rnn,
                             monitor='val_acc', verbose=1, save_best_only=True, mode='max')

encoder_log = autoencoder.fit(train_X, train_X,
                              batch_size=BATCH_SIZE,
                              validation_split=0.2,
                              callbacks=[checkpoint],
                              epochs=N_ITER)

#
#   Model   -   Autoencoder (Plot)
#

plt.title('model loss')
legend_names = []
# summarize history for accuracy
plt.plot(encoder_log.history['loss'])
plt.plot(encoder_log.history['val_loss'])
legend_names.extend(['train', 'validation'])

plt.legend(legend_names, loc='upper left')

plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()

encoded_train_X = encoder.predict(train_X)
encoded_test_X = encoder.predict(test_X)

fig1 = info_SP500.plot()
fig1.show()

#
#   Model   -   Attention
#

BATCH_SIZE = 10
N_ITER = 25

from tensorflow.keras.callbacks import ModelCheckpoint
from tzeentch.models.model_factories import make_attention_model

model = make_attention_model(encoded_train_X.shape[1:], 3)

checkpoint = ModelCheckpoint(path_to_best_att,
                             monitor='val_acc', verbose=1, save_best_only=True, mode='max')

attention_log = model.fit(encoded_train_X, tensorflow.keras.utils.to_categorical(train_Y, num_classes=None),
                              batch_size=BATCH_SIZE,
                                validation_split=0.2,
                              callbacks=[checkpoint],
                              epochs=N_ITER)

#
#   Model   -   Attention (Plot)
#

plt.title('model loss')
legend_names = []
# summarize history for accuracy
plt.plot(attention_log.history['loss'])
plt.plot(attention_log.history['val_loss'])
legend_names.extend(['train', 'validation'])

plt.legend(legend_names, loc='upper left')

plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()

#
#   Simluation
#

import importlib

importlib.reload(tzeentch.simulation.simulators)
from tzeentch.simulation.simulators import simulate_vshold
model.load_weights(path_to_best_att)
predictions = model.predict(encoded_test_X)

sim_model, sim_benchmark, sim_decisions, sim_best = simulate_vshold(
        seq_len=FUTURE_PERIOD_PREDICT,
        close_col='close',
        predictions=predictions,
        df_historical_data=panel_df_test_full)

plt.title(target_columns[0])
plt.plot(sim_model, label='prediction return')
plt.plot(sim_benchmark, label='benchmark return')
plt.xticks(rotation=30)
plt.legend()
plt.show()

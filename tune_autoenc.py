import os

import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress tf warnings

# imports a module for calculating with dates
import datetime as dt
import matplotlib.pyplot as plt

import pandas as pd
import tensorflow.python.keras.layers

# PARAMETERS

START = dt.datetime.fromisocalendar(2005, 1, 1).date()  # has to be the 1st of January of some year
END = dt.datetime.fromisocalendar(2021, 1, 1).date()

# FILES
path_to_best_encoder = os.path.join("models", "autoencoder-lstm_stocks", "val_acc-optimized_enc")
path_to_best_rnn = os.path.join("models", "autoencoder-lstm_stocks", "val_acc-optimized_rnn")
path_to_best_att = os.path.join("models", "autoencoder-lstm_stocks", "val_acc-optimized_att")

# TUNING PARAMETERS
MAX_TUNING_EPOCHS_AENC = 50

path_to_hp_encoder = os.path.join("models", "HPS", "autoencoder", f"hyperparameters_max{MAX_TUNING_EPOCHS_AENC}.json")

# this imports support for hinting types in methods (only interesting for software)


# period model shall predict into future
FUTURE_PERIOD_PREDICT = 31

# input and target columns for the autoencoder model
input_columns_autoenc = ['open', 'high', 'low', 'close', 'volume', 'trend_cci', 'momentum_stoch',
                         'trend_ema20', 'trend_vortex_ind_pos', 'trend_vortex_ind_neg',
                         'trend_vortex_ind_diff',
                         'adjclose', 'trend_sma_fast', 'trend_sma_slow', 'trend_stc',
                         'trend_macd', 'volatility_bbh', 'volatility_bbm', 'volatility_dcl', 'volatility_bbl',
                         'momentum_rsi',
                         'volume_cmf'
                         ]

# has to be even
target_columns = ['close']

training_handles = [#'^IXIC', '^GDAXI', '^N225',
                    '^GSPC']


for handle in training_handles:
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
    handle_informations: IndexInfo = cast(IndexInfo, DataSource.retrieve_yfinance(handle, start=START, end=END))

    #
    # Preprocessing - reindexing and classifying dataset
    #

    # renmae and classify
    from tzeentch.preprocessing.noise_filters import extract_and_preapre_features

    input_df, input_df_only_renamed = extract_and_preapre_features(seq_len=FUTURE_PERIOD_PREDICT,
                                                                   stock_info=handle_informations,
                                                                   feature_colums=input_columns_autoenc,
                                                                   target_columns=target_columns)

    # apply scaling
    from tzeentch.preprocessing.noise_filters import apply_min_max_scaling

    panel_df_train, panel_df_test = apply_min_max_scaling(input_df, target_columns)
    panel_df_test_full = input_df_only_renamed[~input_df_only_renamed.index.isin(panel_df_train.index)]

    # de-noise using wavelet transforms
    from tzeentch.transformer.sequence_transformers import sequence_generator

    SEQ_LEN = FUTURE_PERIOD_PREDICT

    train_X, train_Y = sequence_generator(panel_df_train, SEQ_LEN, shuffle=True, seed=101)
    test_X, test_Y = sequence_generator(panel_df_test, SEQ_LEN, shuffle=False)

    #
    #   Training -   Autoencode
    #

    BATCH_SIZE = 10

    from tensorflow.keras.callbacks import ModelCheckpoint

    assert (FUTURE_PERIOD_PREDICT, len(input_columns_autoenc)) == (train_X.shape[1:]), \
        f"input_columns_autoenc should be even lengthened " \
        f"but is {(FUTURE_PERIOD_PREDICT, len(input_columns_autoenc))} != {train_X.shape[1:]}"

    checkpoint = ModelCheckpoint(path_to_best_encoder,
                                 monitor='val_acc', verbose=1, save_best_only=True, mode='max')


    #
    #       Model Autoencoder - Hyperparameter Tuning
    #

    import keras_tuner as kt

    from tzeentch.models.model_factories import make_autoencoder_model



    def wrapper_autoencoder(hp):
        encoder, autoencoder = make_autoencoder_model(
                (FUTURE_PERIOD_PREDICT, len(input_columns_autoenc)), len(input_columns_autoenc), hp=hp
        )

        return autoencoder


    tuner = kt.Hyperband(
            wrapper_autoencoder,
            objective='val_accuracy',
            max_epochs=MAX_TUNING_EPOCHS_AENC,
            hyperband_iterations=3)

    tuner.search(train_X, train_X,
                 validation_split=0.2,
                 epochs=50,
                 callbacks=[tensorflow.keras.callbacks.EarlyStopping(patience=5)])

    with open(path_to_hp_encoder, 'w') as f:
        import json
        config = tuner.get_best_hyperparameters(1)[0].get_config()
        json.dump(config, f)


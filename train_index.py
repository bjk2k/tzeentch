import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress tf warnings

# imports a module for calculating with dates
import datetime as dt
import matplotlib.pyplot as plt

import json
import keras_tuner as kt
import tensorflow.python.keras.layers

# PARAMETERS

START = dt.datetime.fromisocalendar(2005, 1, 1).date()  # has to be the 1st of January of some year
END = dt.datetime.fromisocalendar(2021, 1, 1).date()

# FILES
path_to_best_encoder = os.path.join("models", "autoencoder-lstm_index", "val_acc-optimized_enc")
path_to_best_rnn = os.path.join("models", "autoencoder-lstm_index", "val_acc-optimized_rnn")
path_to_best_att = os.path.join("models", "autoencoder-lstm_index", "val_acc-optimized_att")

# this imports support for hinting types in methods (only interesting for software)

# loading best hyperparameters
MAX_TUNING_EPOCHS_AENC = 50

path_to_hp_encoder = os.path.join(
        "models", "HPS", "autoencoder", f"hyperparameters_max{MAX_TUNING_EPOCHS_AENC}_best.json"
)
path_to_hp_attention = os.path.join(
        "models", "HPS", "attention", f"hyperparameters_max{MAX_TUNING_EPOCHS_AENC}_best.json"
)

with open(path_to_hp_encoder, 'r') as f:
    saved_config = json.load(f)

with open(path_to_hp_attention, 'r') as f:
    saved_config_att = json.load(f)

encoder_parameters = kt.HyperParameters.from_config(saved_config)
attention_parameters = kt.HyperParameters.from_config(saved_config_att)

# period model shall predict into future
FUTURE_PERIOD_PREDICT = 31

# input and target columns for the autoencoder model
input_columns_autoenc = ['open', 'high', 'low', 'close', 'volume', 'trend_cci', 'momentum_stoch',
                         'trend_ema20',
                         'adjclose', 'trend_sma_fast', 'trend_sma_slow',
                         'trend_macd', 'volatility_bbh', 'volatility_bbm', 'volatility_dcl', 'volatility_bbl',
                         'momentum_rsi',
                         'volume_cmf'
                         ]

# has to be even
target_columns = ['close']

training_handles = ['^GSPC']

#
#   Model Definitions
#
from tzeentch.models.model_factories import make_autoencoder_model
from tzeentch.models.model_factories import make_attention_model

encoder, autoencoder = make_autoencoder_model(
        (FUTURE_PERIOD_PREDICT, len(input_columns_autoenc)), len(input_columns_autoenc), hp=encoder_parameters
)
attention_model = make_attention_model(
        (FUTURE_PERIOD_PREDICT, len(input_columns_autoenc)), 3, hp=attention_parameters
)

for handle in training_handles:
    #
    #   Information Retrieval
    #
    from typing import cast

    import tzeentch.stockwrappers

    from tzeentch.stockwrappers import DataSource
    from tzeentch.stockwrappers import IndexInfo

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
                                 monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

    encoder_log = autoencoder.fit(train_X, train_X,
                                  batch_size=BATCH_SIZE,
                                  validation_split=0.2,
                                  callbacks=[checkpoint],
                                  epochs=encoder_parameters.values.get('tuner/epochs'))

    autoencoder.load_weights(path_to_best_encoder)

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

    #
    #   Model   -   Attention
    #

    BATCH_SIZE = 60

    from tensorflow.keras.callbacks import ModelCheckpoint
    from tzeentch.callbacks.pushbullet_callback import NotificationCallback

    checkpoint = ModelCheckpoint(path_to_best_att,
                                 monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    attention_log = attention_model.fit(encoded_train_X,
                                        tensorflow.keras.utils.to_categorical(train_Y, num_classes=None),
                                        batch_size=BATCH_SIZE,
                                        validation_split=0.2,
                                        callbacks=[checkpoint],
                                        epochs=attention_parameters.values.get('tuner/epochs'))

    #
    #   Model   -   Attention (Plot)
    #

    plt.title('model loss')
    legend_names = []
    # summarize history for accuracy
    plt.plot(attention_log.history['loss'])
    plt.plot(attention_log.history['val_loss'])
    plt.plot(attention_log.history['categorical_crossentropy'])

    legend_names.extend(['train', 'validation'])

    plt.legend(legend_names, loc='upper left')

    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()

    #
    #   Simluation
    #

    from tzeentch.simulation.simulators import simulate_vshold

    attention_model.load_weights(path_to_best_att)
    predictions = attention_model.predict(encoded_test_X)

    sim_model, sim_benchmark, sim_decisions, sim_best = simulate_vshold(
            seq_len=SEQ_LEN,
            close_col='close',
            predictions=predictions,
            df_historical_data=panel_df_test_full)

    plt.title(f"{handle} - {target_columns[0]}")
    plt.plot(sim_model, label='prediction return')
    plt.plot(sim_benchmark, label='benchmark return')
    plt.xticks(rotation=30)
    plt.legend()
    plt.show()

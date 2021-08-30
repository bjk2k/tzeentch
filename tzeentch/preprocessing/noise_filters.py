import pandas as pd
from typing import List
from tzeentch.stockwrappers import StockInfo


def extract_and_preapre_features(seq_len: int,
                        stock_info: StockInfo,
                        feature_colums: List[str],
                        target_columns: List[str], significant_criteria: float = 0.3):
    input_df: pd.DataFrame = stock_info.technical_indicators.copy(deep=True).get(feature_colums)

    # remove index and turn it into normal date
    input_df.dropna(inplace=True)
    input_df = input_df.apply(pd.to_numeric)

    input_df.reset_index(inplace=True)
    input_df['timestamp'] = input_df['index'].apply(lambda x: x.timestamp())
    input_df.drop(columns='index', inplace=True)

    # create target column (return if hold for FUTURE_PERIOD_PREDICT segments)
    for col in target_columns:
        input_df[col + '_target'] = input_df[col].shift(-seq_len)
        input_df[col + '_target_return'] = (input_df[col + '_target'] - input_df[col]) / input_df[col]

    input_df_old = input_df.copy(deep=True)

    # classify target column into 0,1,2
    # (wether target column went significally up/down/neither during FUTURE_PERIOD_PREDICT seg)

    import numpy as np

    def classify_trinary(values):
        gp_std = np.std(values)

        target = []
        for value in values:
            if significant_criteria * gp_std < value:
                target.append(2)
            elif -significant_criteria * gp_std > value:
                target.append(0)
            else:
                target.append(1)

        return pd.Series(target)

    for col in target_columns:
        input_df[col + '_target'] = input_df[col + '_target_return'].transform(classify_trinary)

    return input_df, input_df_old


def apply_min_max_scaling(input_df: pd.DataFrame, target_columns: List[str]):
    #
    #   Preprocessing - scaling and denoising
    #

    # min max scale certain columns
    from sklearn.preprocessing import minmax_scale

    excluded_columns = []
    columns_not_to_scale = ['timestamp', 'target']

    for col in input_df.columns:
        dont_scale = False

        for dont_scale_column in columns_not_to_scale:
            if dont_scale_column in col:
                dont_scale = True
                excluded_columns.append(col)

        if dont_scale: continue

        input_df[col] = minmax_scale(input_df[col].values)

    target_columns = [col + "_target" for col in target_columns]

    for col in target_columns:
        excluded_columns.remove(col)

    input_df = input_df.filter(regex="^(?!({0})$).*$".format('|'.join(excluded_columns)))

    # split train test dataset
    TRAIN_RATIO = 0.8

    panel_df_train = input_df.iloc[:int(input_df.shape[0] * TRAIN_RATIO)]
    panel_df_test = input_df[~input_df.index.isin(panel_df_train.index)]

    return panel_df_train, panel_df_test

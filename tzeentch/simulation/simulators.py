from typing import List

import numpy as np
import pandas as pd

def simulate_vshold(
        seq_len: int,
        close_col: str,
        predictions: List[int],
        df_historical_data: pd.DataFrame,
        benchmark_start: int = 100,
        prediction_start: int = 100):
    benchmark_hist = []
    benchmark_return = benchmark_start

    prediction_hist = []
    prediction_return = prediction_start

    hold = True
    hold_hist = []
    best = 100
    best_hist = []

    df_historical_data[close_col + '_target_return'] = df_historical_data[close_col + '_target'].pct_change()
    for prediction, (i, r) in zip(predictions, df_historical_data.iloc[seq_len - 1:].iterrows()):

        if hold:
            prediction_return *= (1 + r[close_col + '_target_return'])

        best_hist.append(best)

        benchmark_return *= (1 + r[close_col + '_target_return'])

        benchmark_hist.append(benchmark_return)
        prediction_hist.append(prediction_return)

        hold = (np.argmax(prediction) == 2)
        hold_hist.append(hold)

    return prediction_hist, benchmark_hist, hold_hist, best_hist
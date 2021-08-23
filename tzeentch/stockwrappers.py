"""tzeentch main file"""
import datetime as dt

import yfinance as yf
import yahoo_fin.stock_info as si

import pandas as pd
import numpy as np

from dataclasses import dataclass, field

from typing import Optional, Union, Tuple

from ta import add_all_ta_features


@dataclass(frozen=True)
class StockInfo:
    """simple wrapper for stock information and later evaluation"""
    handle: str
    start: dt.date
    end: dt.date

    present: pd.DataFrame = field(init=False)
    ticker: yf.Ticker = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, 'ticker', yf.Ticker(self.handle))
        object.__setattr__(self, 'present', yf.download(self.handle, self.start, self.end))


def calculate_technical_ind_index(handle: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    """fetches the technical indicators for a given Index

    Returns:

    """
    toReturn: pd.DataFrame = si.get_data(handle).loc[start - dt.timedelta(days=365): end]

    toReturn = add_all_ta_features(
            toReturn, open="open", high="high", low="low", close="adjclose", volume="volume")

    #add ema20

    from ta.trend import ema_indicator

    indicator_ema20 = ema_indicator(toReturn['close'], window=20, fillna=True)

    return toReturn.loc[start: end]


@dataclass(frozen=True)
class IndexInfo(StockInfo):
    """simple wrapper for stock information and later evaluation"""
    technical_indicators: pd.DataFrame = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, 'technical_indicators', calculate_technical_ind_index(self.handle, self.start, self.end))
        super().__post_init__()


class DataSource:
    """simple wrapper for yfinance stock information evaluation"""

    index_handlers = ['^GSPC']

    @staticmethod
    def retrieve_yfinance(stock_handle: str, start: Optional[dt.date] = None, end: Optional[dt.date] = None, n=14) \
            -> Union[StockInfo, IndexInfo]:
        """retrieves stock information form yfinance and creates a :class`StockInfo` dataclass from it

        Returns:
            dataclass holding current stock information
        """

        if "^" in stock_handle:
            _ = IndexInfo(
                    handle=stock_handle,
                    start=start,
                    end=end)
            print("[<] fetched index information ...")
            return _

        _ = IndexInfo(
                handle=stock_handle,
                start=start,
                end=end)
        print("[<] fetched stock information ...")

        return _

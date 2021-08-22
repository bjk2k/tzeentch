"""tzeentch main file"""
import datetime as dt

import yfinance as yf

import pandas as pd
import numpy as np

from dataclasses import dataclass

from typing import Optional, Union, Tuple

from tensorflow import keras as ks


@dataclass
class StockInfo:
    """simple wrapper for stock information and later evaluation"""
    ticker: yf.Ticker
    present: pd.DataFrame

    start: dt.date
    end: dt.date


@dataclass
class IndexInfo(StockInfo):
    """simple wrapper for stock information and later evaluation"""

    ext_present: pd.DataFrame

    @property
    def n(self) -> int:
        return self.ext_present.shape[0] - self.present.shape[0]

    @property
    def macd(self) -> pd.DataFrame:
        exp1 = self.present['Adj Close'].ewm(span=12, adjust=False).mean()
        exp2 = self.present['Adj Close'].ewm(span=26, adjust=False).mean()
        macd: pd.Series = exp1 - exp2

        return macd.to_frame()

    @property
    def mad(self) -> pd.DataFrame:
        return self.typical_price.rolling(self.n).apply(lambda x: pd.Series(x).mad())

    @property
    def typical_price(self) -> pd.DataFrame:
        series: pd.Series = ((self.ext_present['High'] + self.ext_present['Low'] + self.ext_present['Close']) / 3)

        return series.to_frame()

    @property
    def sma(self) -> pd.DataFrame:
        return self.typical_price.rolling(self.n).mean()

    @property
    def cci(self) -> pd.DataFrame:
        return (self.typical_price - self.sma) / (self.mad * 0.015)

    @property
    def atr(self) -> pd.DataFrame:
        high_low = self.ext_present['High'] - self.ext_present['Low']
        high_close = np.abs(self.ext_present['High'] - self.ext_present['Close'].shift())
        low_close = np.abs(self.ext_present['Low'] - self.ext_present['Close'].shift())

        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)

        atr = true_range.rolling(14).sum() / 14

        return atr

    @property
    def BOLL(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        sma = self.ext_present['Close'].rolling(self.n).mean()
        std = self.ext_present['Close'].rolling(self.n).std()
        bollinger_up: pd.Series = sma + std * 2  # Calculate top band
        bollinger_down: pd.Series = sma - std * 2  # Calculate bottom band

        return bollinger_up.to_frame().loc[self.start: self.end], bollinger_down.to_frame().loc[self.start: self.end]

    def _calculate_ema(self, days, smoothing=2) -> pd.Series:
        prices = self.ext_present['Close']
        ema = [sum(prices[:days]) / days]
        for price in prices[days:]:
            ema.append((price * (smoothing / (1 + days))) + ema[-1] * (1 - (smoothing / (1 + days))))
        return pd.Series(ema[1:])

    @property
    def EMA14(self) -> pd.DataFrame:
        toReturn = self._calculate_ema(14).to_frame()
        toReturn.index = self.present.index
        return toReturn

    @property
    def technical_data(self) -> pd.DataFrame:
        toReturn: pd.DataFrame = self.ext_present.copy()
        toReturn['MACD'] = self.macd
        toReturn['CCI'] = self.cci
        toReturn['ATR'] = self.atr
        toReturn['BOLL_H'], toReturn['BOLL_L'] = self.BOLL
        toReturn['EMA20'] = self.EMA14

        return toReturn.loc[self.start: self.end]



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
            print("[<] fetched index information ...")
            start_shifted = start - dt.timedelta(days=n)
            return IndexInfo(
                    ticker=yf.Ticker(stock_handle),
                    present=yf.download(stock_handle, start, end),
                    ext_present=yf.download(stock_handle, start_shifted, end),
                    start=start,
                    end=end)

        print("[<] fetched stock information ...")
        return StockInfo(
                ticker=yf.Ticker(stock_handle),
                present=yf.download(stock_handle, start, end),
                start=start,
                end=end)

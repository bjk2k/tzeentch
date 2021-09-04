"""tzeentch main file"""

import datetime as dt
import os

from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import yfinance as yf
import yahoo_fin.stock_info as si

from ta import add_all_ta_features

package_directory = os.path.dirname(os.path.abspath(__file__))
path_to_libor_data = os.path.join(package_directory, 'data', 'LIBOR_USD.csv')


def calculate_technical_ind_index(handle: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    """fetches the technical indicators for a given Index

    Args:
        handle: index or stock handle
        start: start date for requested information
        end: end date for requested information

    Returns:
        a dataframe with lots of technical indicators
    """
    toReturn: pd.DataFrame = si.get_data(handle).loc[start - dt.timedelta(days=365): end]

    with np.errstate(invalid='ignore'):
        toReturn = add_all_ta_features(
                toReturn, open="open", high="high", low="low", close="adjclose", volume="volume")

        # add ema20

        from ta.trend import ema_indicator
        from ta.trend import SMAIndicator

        toReturn['trend_ema20'] = ema_indicator(toReturn['close'], window=20, fillna=True)

        toReturn['trend_ma10'] = SMAIndicator(toReturn['close'], window=5, fillna=True)
        toReturn['trend_ma5'] = SMAIndicator(toReturn['close'], window=5, fillna=True)

    return toReturn.loc[start: end]


def retrieve_macro_economic_info(start: dt.date, end: dt.date):
    """fetches the macro-economic variables

    Args:
        start: start date of the requested information
        end: end date of the requested information

    Returns:
        dataframe from us dollar exchange rates and interbank offered rates in a given time perios

    """
    # fetch current usd index from yahoo finance
    dollar_index: pd.DataFrame = yf.download("DX-Y.NYB", start=start, end=end)
    # drop volume column as it is zero
    dollar_index = dollar_index.drop(columns=['Volume'])
    # add prefix so column name matches the name format in technical indicators frame
    dollar_index = dollar_index.add_prefix('macro_DXY_')

    # fetch the london interchange rates from a csv (needs to be updated later as data source)
    ibor_rates: pd.DataFrame = pd.read_csv(path_to_libor_data, parse_dates=[0], index_col='date')

    dollar_index['macro_LIBOR_USD'] = ibor_rates['macro_libor_usd'].loc[start: end]

    return dollar_index


@dataclass(frozen=True)
class StockInfo:
    """simple wrapper for stock information and later evaluation"""
    handle: str
    start: dt.date
    end: dt.date

    present: pd.DataFrame = field(init=False)
    ticker: yf.Ticker = field(init=False)

    technical_indicators: pd.DataFrame = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, 'ticker', yf.Ticker(self.handle))
        object.__setattr__(self, 'present', yf.download(self.handle, self.start, self.end))
        object.__setattr__(self, 'technical_indicators',
                           calculate_technical_ind_index(self.handle, self.start, self.end))

    def plot(self) -> go.Figure:
        """returns a plotly.graph_object figure showing the historic ohlcv data contained

        Returns:
            said thing
        """
        historic_daily_ohlcv = self.present

        for i in ['Open', 'High', 'Close', 'Low']:
            historic_daily_ohlcv[i] = historic_daily_ohlcv[i].astype('float64')

        fig = go.Figure(data=go.Ohlc(x=historic_daily_ohlcv.index,
                                     open=historic_daily_ohlcv['Open'],
                                     high=historic_daily_ohlcv['High'],
                                     low=historic_daily_ohlcv['Low'],
                                     close=historic_daily_ohlcv['Close'],
                                     name='OHLC(V)'
                                     )
                        )

        fig.update_layout(
                title=f"Historic OHLC(V) data of {self.handle} from {self.start} till {self.end}",
        )

        return fig


@dataclass(frozen=True)
class IndexInfo(StockInfo):
    """simple wrapper for stock information and later evaluation"""

    macro_indicators: pd.DataFrame = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self, 'macro_indicators',
                           retrieve_macro_economic_info(self.start, self.end))
        object.__setattr__(self, 'technical_indicators',
                           self.technical_indicators.join(self.macro_indicators))


@dataclass(frozen=True)
class CurrencyInfo(StockInfo):
    """simple wrapper for currency information and later evaluation"""

    def __post_init__(self):
        super().__post_init__()


class DataSource:
    """simple wrapper for yfinance stock information evaluation"""

    index_handlers = ['^GSPC']

    @staticmethod
    def retrieve_yfinance(stock_handle: str, start: Optional[dt.date] = None, end: Optional[dt.date] = None) \
            -> Union[StockInfo, IndexInfo]:
        """retrieves stock information form yfinance and creates a :class`StockInfo` dataclass from it

        Returns:
            dataclass holding current stock information
        """

        if "^" in stock_handle or "INDEX" in stock_handle:
            print(f"[>] requesting index information for [{stock_handle}] ...")
            _ = IndexInfo(
                    handle=stock_handle,
                    start=start,
                    end=end)
            print(f"[<] fetched index information for [{stock_handle}] ...\n")
            return _

        print(f"[>] requesting stock information [{stock_handle}] ...")
        _ = IndexInfo(
                handle=stock_handle,
                start=start,
                end=end)
        print(f"[<] fetched stock information [{stock_handle}] ...\n")

        return _

    @staticmethod
    def retrieve_crypto(stock_handle: str, start: Optional[dt.date] = None, end: Optional[dt.date] = None) \
            -> Union[StockInfo, IndexInfo]:
        """retrieves stock information form yfinance and creates a :class`StockInfo` dataclass from it

        Returns:
            dataclass holding current stock information
        """

        print("[>] requesting stock information ...")
        _ = IndexInfo(
                handle=stock_handle,
                start=start,
                end=end)
        print("[<] fetched stock information ...\n")

        return _

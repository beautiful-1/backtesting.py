"""Data and utilities for testing."""
import pandas as pd


def _read_file(filename):
    from os.path import dirname, join

    return pd.read_csv(join(dirname(__file__), filename),
                       index_col=0, parse_dates=True, infer_datetime_format=True)


GOOG = _read_file('GOOG.csv')
"""DataFrame of daily NASDAQ:GOOG (Google/Alphabet) stock price data from 2004 to 2013."""

EURUSD = _read_file('EURUSD.csv')

btcusdt = _read_file('database_db_dbbardata.csv')
"""DataFrame of hourly EUR/USD forex data from April 2017 to February 2018."""

"""
arr: pd.Series：这是一个 Pandas Series 对象，包含了要计算移动平均的数据。Pandas Series 是 Pandas 库中的一种数据结构，类似于一维数组，但具有附加的标签和功能。

n: int：这是一个整数，表示要计算的移动平均的期间（窗口大小）。移动平均是在滑动窗口内计算的，窗口的大小由 n 指定。
"""


def SMA(arr: pd.Series, n: int) -> pd.Series:
    """
    Returns `n`-period simple moving average of array `arr`.
    """
    return pd.Series(arr).rolling(n).mean()

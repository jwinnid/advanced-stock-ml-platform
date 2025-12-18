# indicators.py
import pandas as pd
import numpy as np

def ema(series: pd.Series, span: int):
    return series.ewm(span=span, adjust=False).mean()

def bollinger(series: pd.Series, window: int=20, num_std: int=2):
    m = series.rolling(window).mean()
    s = series.rolling(window).std()
    u = m + num_std*s
    l = m - num_std*s
    return m, u, l

def rsi(series: pd.Series, period: int=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100/(1+rs))

def macd(series: pd.Series, fast=12, slow=26, signal=9):
    f = series.ewm(span=fast).mean()
    s = series.ewm(span=slow).mean()
    macd_line = f - s
    signal_line = macd_line.ewm(span=signal).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def stochastic(df: pd.DataFrame, k_period=14, d_period=3):
    low = df['low'].rolling(k_period).min()
    high = df['high'].rolling(k_period).max()
    k = 100 * (df['close'] - low) / (high - low + 1e-9)
    d = k.rolling(d_period).mean()
    return k, d

def add_all(df: pd.DataFrame):
    df = df.copy()
    df['ema_12'] = ema(df['close'], 12)
    df['ema_26'] = ema(df['close'], 26)
    df['rsi_14'] = rsi(df['close'])
    macd_line, macd_sig, macd_hist = macd(df['close'])
    df['macd'] = macd_line
    df['macd_signal'] = macd_sig
    df['macd_hist'] = macd_hist
    mid, up, low = bollinger(df['close'])
    df['bb_mid'] = mid
    df['bb_upper'] = up
    df['bb_lower'] = low
    k, d = stochastic(df)
    df['stoch_k'] = k
    df['stoch_d'] = d
    return df

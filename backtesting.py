# backtesting.py
import pandas as pd
import numpy as np

def compute_sharpe(returns: pd.Series, risk_free=0.0):
    # daily to annual
    mean = returns.mean() * 252
    std = returns.std() * (252 ** 0.5)
    if std == 0:
        return 0.0
    return (mean - risk_free) / std

def max_drawdown(series: pd.Series):
    roll_max = series.cummax()
    drawdown = (series - roll_max) / roll_max
    return abs(drawdown.min())

def run_backtest(df: pd.DataFrame, signals: pd.Series, initial_capital=10000.0):
    df = df.copy().reset_index(drop=True)
    signals = signals.reset_index(drop=True).fillna(0)
    df['signal'] = signals
    df['next_close'] = df['close'].shift(-1)
    df['pct'] = df['next_close'] / df['close'] - 1
    df['strategy_ret'] = df['signal'] * df['pct']
    df['strategy_ret'] = df['strategy_ret'].fillna(0)
    df['portfolio'] = initial_capital * (1 + df['strategy_ret']).cumprod()
    daily = df['strategy_ret'].fillna(0)
    sharpe = compute_sharpe(daily)
    vol = daily.std() * (252 ** 0.5)
    maxdd = max_drawdown(df['portfolio'])
    metrics = {"sharpe":float(sharpe),"volatility":float(vol),"max_drawdown":float(maxdd)}
    return df, metrics

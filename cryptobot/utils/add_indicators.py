# add_indicators.py
from __future__ import annotations
import pandas as pd

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add a standard set of indicators to OHLCV DataFrame."""
    if df.empty:
        return df

    # EMA
    for p in (9, 21, 50):
        df[f"ema_{p}"] = df["close"].ewm(span=p, adjust=False).mean()

    # SMA
    for p in (20, 200):
        df[f"sma_{p}"] = df["close"].rolling(window=p).mean()

    # RSI
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, 1)
    df["rsi"] = 100 - (100 / (1 + rs))

    # ATR
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = high_low.combine(high_close, max).combine(low_close, max)
    df["atr"] = tr.rolling(window=14).mean()

    # MACD
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    return df

"""Quickstart feature functions. A feature function takes RAW rows and returns features aligned to them."""
import pandas as pd


def features(df):
    """Looks harmless, but normalises with the mean and std of the WHOLE file: row 10 'knows' the future."""
    c = df["Close"]
    return pd.DataFrame({
        "sma20": c.rolling(20).mean(),
        "zscore": (c - c.mean()) / c.std(),
    })


def features_fixed(df):
    """Same idea using only past data (expanding window)."""
    c = df["Close"]
    return pd.DataFrame({
        "sma20": c.rolling(20).mean(),
        "zscore": (c - c.expanding(20).mean()) / c.expanding(20).std(),
    })

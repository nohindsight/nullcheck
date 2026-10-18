"""Synthetic fixtures: fictional prices plus planted-leak and causal feature functions. Not market data."""
import numpy as np
import pandas as pd


def make_ohlcv(n=600, seed=0, drift=0.0004, vol=0.012, gaps=0):
    """Fictional daily business-day OHLCV from a seeded random walk. gaps>0 blanks that many short Volume runs."""
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(drift, vol, n)))
    open_ = close * np.exp(rng.normal(0, vol / 3, n))
    high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, vol / 2, n)))
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, vol / 2, n)))
    volume = rng.integers(1_000, 50_000, n).astype(float)
    for _ in range(gaps):
        s = int(rng.integers(5, n - 10))
        volume[s:s + int(rng.integers(1, 4))] = np.nan
    idx = pd.bdate_range("2020-01-01", periods=n, name="Date")
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume}, index=idx)

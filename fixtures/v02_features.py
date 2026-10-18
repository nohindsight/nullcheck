"""Disclosed blind-suite misses M1--M5, with named functions for replay."""
import numpy as np
import pandas as pd


def prices(n=600, seed=1):
    idx = pd.bdate_range("2021-01-04", periods=n, name="Date")
    c = 100 * np.exp(np.cumsum(np.random.default_rng(seed).normal(0, 0.01, n)))
    return pd.DataFrame({"Open": c, "High": c * 1.01, "Low": c * .99, "Close": c, "Volume": 1e4}, index=idx)


def warmup_bfill(d):
    return pd.DataFrame({"sma": d.Close.rolling(20).mean().bfill()})


def zero_bfill(d):
    return pd.DataFrame({"c": d.Close.replace(0., np.nan).bfill()})


def halt_bfill(d):
    return pd.DataFrame({"c": d.Close.where(d.Volume != 0).bfill()})


def month_end_future(d):
    me = (d.index + pd.offsets.BMonthEnd(0)) == d.index
    return pd.DataFrame({"px": d.Close.where(~me, d.Open.shift(-1))})


def descending_lag(d):
    return pd.DataFrame({"lag1": d.Close.shift(1)})


def descending_causal(d):
    return pd.DataFrame({"sma": d.sort_index().Close.rolling(5).mean()}).reindex(d.index)


def mixed_random(d):
    return pd.DataFrame({"z": (d.Close-d.Close.mean())/d.Close.std(), "rnd": np.random.rand(len(d))}, index=d.index)


def mixed_text(d):
    return pd.DataFrame({"z": (d.Close-d.Close.mean())/d.Close.std(), "sym": "ACME"}, index=d.index)


def sparse_clean(d):
    return pd.DataFrame({"sma": d.Close.rolling(20).mean(), "signal": d.Close.where(d.Close.diff() > 1),
                         "empty": np.nan}, index=d.index)


def long_warmup(d):
    return d.Close.rolling(200).mean()


def leak_with_inf(d):
    return pd.DataFrame({"z": d.Close-d.Close.mean(), "bad": np.inf}, index=d.index)

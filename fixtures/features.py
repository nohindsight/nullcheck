"""Feature functions for the validation suite. LEAKS use information from later rows; CLEAN ones do not.
Each takes raw OHLCV rows and returns features aligned to them."""
import numpy as np
import pandas as pd


# ---- planted leaks -------------------------------------------------------------------------------------------
def leak_next_bar(df):            # next bar's close compared with this one
    c = df["Close"]
    return pd.DataFrame({"up_next": (c.shift(-1) > c).astype(float)})


def leak_centred_mean(df):        # centred window uses 10 future rows
    c = df["Close"]
    return pd.DataFrame({"dev": c - c.rolling(21, center=True).mean()})


def leak_whole_sample_z(df):      # normalised with the mean/std of the whole file
    c = df["Close"]
    return pd.DataFrame({"z": (c - c.mean()) / c.std()})


def leak_minmax_scale(df):        # scaled with the min/max of the whole file
    c = df["Close"]
    return pd.DataFrame({"scaled": (c - c.min()) / (c.max() - c.min())})


def leak_whole_sample_rank(df):   # percentile rank over the whole file
    return pd.DataFrame({"pct_rank": df["Close"].rank(pct=True)})


def leak_weekly_bfill(df):        # weekly close back-filled onto the days of that week
    c = df["Close"]
    w = c.resample("W-FRI").last()
    return pd.DataFrame({"week_close": w.reindex(c.index, method="bfill")})


def leak_future_label(df):        # a forward 5-day return used as a feature
    return pd.DataFrame({"fwd5": df["Close"].pct_change(5).shift(-5)})


def leak_bfill_gaps(df):          # missing volume back-filled from the next available row
    return pd.DataFrame({"vol": df["Volume"].bfill()})


LEAKS = {f.__name__: f for f in [leak_next_bar, leak_centred_mean, leak_whole_sample_z, leak_minmax_scale,
                                 leak_whole_sample_rank, leak_weekly_bfill, leak_future_label, leak_bfill_gaps]}


# ---- causal controls -----------------------------------------------------------------------------------------
def clean_sma_cross(df):
    c = df["Close"]
    return pd.DataFrame({"fast": c.rolling(10).mean(), "slow": c.rolling(40).mean(),
                         "long": (c.rolling(10).mean() > c.rolling(40).mean()).astype(float)})


def clean_ewm(df):
    c = df["Close"]
    return pd.DataFrame({"ewm_adj": c.ewm(span=20).mean(), "ewm_rec": c.ewm(span=20, adjust=False).mean()})


def clean_expanding_z(df):        # normalised with only past data
    c = df["Close"]
    return pd.DataFrame({"z": (c - c.expanding().mean()) / c.expanding().std()})


def clean_long_warmup(df):        # 200-row window: NaN for most short prefixes
    return pd.DataFrame({"sma200": df["Close"].rolling(200).mean()})


def clean_rsi(df):
    d = df["Close"].diff()
    up, dn = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean(), (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    return pd.DataFrame({"rsi": 100 - 100 / (1 + up / dn)})


def clean_weekly_prev(df):        # last COMPLETED week's close, forward-filled
    c = df["Close"]
    w = c.resample("W-FRI").last().shift(1)
    return pd.DataFrame({"prev_week": w.reindex(c.index, method="ffill")})


def clean_seeded_noise(df):       # seeded randomness drawn in row order: deterministic and causal
    return pd.DataFrame({"noise": np.random.default_rng(7).normal(size=len(df))}, index=df.index)


def clean_drawdown(df):
    c = df["Close"]
    return pd.DataFrame({"dd": c / c.cummax() - 1, "ffill_vol": df["Volume"].ffill()})


WARMUP = {"clean_sma_cross": 40, "clean_ewm": 1, "clean_expanding_z": 2, "clean_long_warmup": 200, "clean_rsi": 15,
          "clean_weekly_prev": 15, "clean_seeded_noise": 0, "clean_drawdown": 0}

CLEAN = {f.__name__: f for f in [clean_sma_cross, clean_ewm, clean_expanding_z, clean_long_warmup, clean_rsi,
                                 clean_weekly_prev, clean_seeded_noise, clean_drawdown]}


# ---- neither: nondeterministic ---------------------------------------------------------------------------------
def unseeded_noise(df):
    return pd.DataFrame({"noise": np.random.default_rng().normal(size=len(df))}, index=df.index)

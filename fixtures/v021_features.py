"""Chronology repros from an internal v0.2.0 review (B2), with named replayable functions."""
import pandas as pd

def lag1(d):
    return pd.DataFrame({"lag1": d.Close.shift(1)}, index=d.index)

def sorted_sma(d):
    return pd.DataFrame({"sma": d.sort_values("Date").Close.rolling(5).mean()}).reindex(d.index)

def identity(d):
    return d.Close

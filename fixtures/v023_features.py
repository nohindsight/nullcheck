"""Minimum coverage regressions; synthetic inputs, no trading performance claims."""
import numpy as np

def passthrough(rows): return rows.Close

def constant(rows): return np.full(len(rows), 7.)

def counters(rows):
    return {"zero": np.zeros(len(rows)), "counter": 2 + 3*np.arange(len(rows))}

def nonlinear_counter(rows): return np.sin(np.arange(len(rows)))

def causal(rows): return rows.Close.rolling(3, min_periods=1).mean()

def causal_mixed(rows):
    return {"raw": rows.Close, "constant": np.ones(len(rows)), "counter": np.arange(len(rows)), "signal": causal(rows)}

def constant_and_leak(rows):
    return {"constant": np.ones(len(rows)), "future": rows.Close.shift(-1)}

def dependency_failure(rows):
    if rows.Close.iloc[0] > rows.Close.iloc[-1]:
        raise ValueError("dependency probe unsupported")
    return causal(rows)

def prepare_copy(rows):
    rows["prediction"] = rows.Close
    return rows


def sparse_events(rows): return rows.Close.diff().gt(0).replace(False, np.nan).astype(float)

def generated_sparse(rows): return np.where(np.arange(len(rows)) % 3, np.nan, 1.)

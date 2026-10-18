"""Synthetic disclosed regression cases for external-input coverage."""
from pathlib import Path
import numpy as np
import pandas as pd

PRECOMPUTED = np.arange(1000.)
FILE = None

def external_global(rows):
    return PRECOMPUTED[:len(rows)]

def external_file(rows):
    return pd.read_csv(FILE)["prediction"].to_numpy()[:len(rows)]

def external_file_and_leak(rows):
    return {"fixed": external_file(rows), "leak": rows.Close.shift(-1)}

def prefix_file_only(rows):
    if len(rows) < 20:
        external_file(rows)
    return rows.Close

def all_nan_file(rows):
    external_file(rows)
    return np.full(len(rows), np.nan)

def prepare_file(rows):
    rows["prediction"] = external_file(rows)
    return rows

def identity(rows):
    return rows.Close

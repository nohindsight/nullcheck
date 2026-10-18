"""Internal exact-equality encoding for supported categorical adapter values.

Codes are deterministic 53-bit BLAKE2b fingerprints, exactly representable as floats.
Every code is checked against its canonical value across a callback's full/prefix
runs: a collision raises and leaves the channel unassessed, never silently equal.
Missing values remain NaN. Codes have no numeric meaning and ignore atol/rtol.
"""
from __future__ import annotations

import datetime
import hashlib
import math
import numpy as np
import pandas as pd


class CategoricalValues(np.ndarray):
    """Marker carried to normalisation, including slices of two-dimensional arrays."""


def categorical(value):
    if value is None or value is pd.NA or value is pd.NaT or isinstance(value,
            (str, bytes, np.str_, np.bytes_, np.datetime64, pd.Timestamp, datetime.date)):
        return True
    if isinstance(value, (pd.Categorical, pd.CategoricalIndex)):
        return True
    if isinstance(value, pd.DataFrame):
        return any(categorical(value[c]) for c in value.columns)
    if isinstance(value, pd.Series):
        if isinstance(value.dtype, (pd.CategoricalDtype, pd.StringDtype)) or pd.api.types.is_datetime64_any_dtype(value.dtype):
            return True
        if pd.api.types.is_numeric_dtype(value.dtype):
            return False  # preserve numeric/nullable-boolean normalization and tolerance
        value = value.to_numpy()
    if isinstance(value, (np.ndarray, list, tuple)):
        arr = np.asarray(value)
        if arr.dtype.kind in "USM":
            return True
        if arr.dtype.kind == "O" and arr.size:
            # Unsupported objects are rejected by canonical(), never repr()'d.
            return any(categorical(x) for x in arr.flat if not isinstance(x, (list, tuple, np.ndarray)))
    return False


def canonical(value):
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (np.datetime64, pd.Timestamp, datetime.datetime)):
        stamp = pd.Timestamp(value)
        return None if pd.isna(stamp) else (b"time:naive:" if stamp.tz is None else b"time:aware:") + str(stamp.value).encode("ascii")
    if isinstance(value, datetime.date):
        return b"date:" + value.isoformat().encode("ascii")
    if isinstance(value, (str, np.str_)):
        return b"str:" + str(value).encode("utf-8", errors="surrogatepass")
    if isinstance(value, (bytes, np.bytes_)):
        return b"bytes:" + bytes(value)
    if isinstance(value, (bool, np.bool_)):
        return b"bool:1" if value else b"bool:0"
    if isinstance(value, (int, np.integer)):
        return b"number:" + str(int(value)).encode("ascii") + b"/1"
    if isinstance(value, (float, np.floating)):
        if math.isnan(value):
            return None
        if not math.isfinite(value):
            raise ValueError("infinite categorical value")
        num, den = value.as_integer_ratio()
        return b"number:" + str(num).encode("ascii") + b"/" + str(den).encode("ascii")
    raise TypeError("unsupported categorical value type: " + type(value).__name__)


def fingerprint(token):
    return int.from_bytes(hashlib.blake2b(token, digest_size=8).digest(), "big") & ((1 << 53) - 1)


class Codec:
    def __init__(self):
        self.encodings = {}
        self.seen = {}

    def convert(self, key, value):
        # Fix the encoding from the full run. Numeric values entering a categorical
        # channel are encoded by value too; a numeric-to-missing transition is NaN.
        encoded = self.encodings.setdefault(key, categorical(value))
        if not encoded:
            return np.nan if value is None or value is pd.NA or value is pd.NaT else value
        if isinstance(value, (pd.Series, pd.DataFrame, pd.Index)):
            arr = value.to_numpy()
        else:
            # NumPy's default coercion would turn (1, "x") into ("1", "x").
            arr = np.asarray(value, dtype=object) if isinstance(value, (list, tuple)) else np.asarray(value)
        result = np.empty(arr.shape, dtype=float).view(CategoricalValues)
        for ix in np.ndindex(arr.shape):
            token = canonical(arr[ix])
            if token is None:
                result[ix] = np.nan
                continue
            code = fingerprint(token)
            old = self.seen.setdefault(code, token)
            if old != token:
                raise ValueError("categorical fingerprint collision; equality not assessed")
            result[ix] = code
        return result

"""backtesting.py adapter. Supported: backtesting 0.6.6 exactly (the only version tested; it relies on the private
Strategy._indicators list and on Backtest.run() calling Strategy.init() on the data it was given).

backtesting.py is labelled AGPL-3.0; its PyPI classifier says "or later". This adapter imports an installed copy and does not bundle or modify it;
whether that is enough for a given distribution is a licensing question this software does not answer.

For each run nullcheck builds a fresh Backtest on the (prepared) rows and runs it only up to the end of
Strategy.init(), then stops before next(). Channels are a snapshot taken at that moment:
  I[i]:name   every array registered with Strategy.I(), in creation order
  attr:name   other numeric attributes set in init(), including private names:
              arrays aligned on the first or last axis; row-length pandas outputs (index checked);
              numeric scalars, bounded fixed-length ndarray/Series state and list/tuple state vectors.
              String/category/datetime values use exact equality; None/NaT remain missing.
              Layout is inferred on the full run and held fixed across shorter prefixes. Unsupported
              attributes are named with their type in limitations and receive unassessed channels.
  data:col    columns that prepare() adds to the raw rows
Anything next() changes later, and all trades, orders and equity, are outside the check.
"""
from __future__ import annotations

import datetime
import warnings
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from ..core import FeatureSnapshot, Result, Settings, check
from ..categorical import Codec, CategoricalValues

OHLCV = ("Open", "High", "Low", "Close", "Volume")
SUPPORTED_VERSIONS = ("0.6.6",)
MAX_STATE_ELEMENTS = 256
MAX_STATE_CELLS = 1_000_000
INIT_STATE_ONLY = "SNAPSHOT_SCOPE=INIT_STATE_ONLY: no computed row-wise channel; only init-state values were compared."


def _state_size(value, budget=MAX_STATE_ELEMENTS, depth=0):
    """Bound before conversion/broadcast, including nested list/tuple arrays."""
    if depth > 8:
        return budget + 1
    if isinstance(value, (np.ndarray, pd.Series, pd.DataFrame, pd.Categorical, pd.Index)):
        return value.size
    if isinstance(value, (list, tuple)):
        size = 0
        for item in value:
            size += _state_size(item, budget - size, depth + 1)
            if size > budget:
                break
        return size
    return 1


class _StopAfterInit(Exception):
    pass


def _backtesting():
    try:
        import backtesting
        from backtesting import Backtest
        from backtesting._util import _Indicator
    except ImportError as e:
        raise ImportError("the backtesting.py adapter needs 'backtesting==0.6.6' installed") from e
    if backtesting.__version__ not in SUPPORTED_VERSIONS:
        raise RuntimeError(f"backtesting {backtesting.__version__} is not supported; tested version: {SUPPORTED_VERSIONS}")
    return backtesting, Backtest, _Indicator


def _snapshot(strat, n: int, _Indicator, layouts: dict, codec=None) -> FeatureSnapshot:
    """Infer layout once on full history, then keep it fixed across prefixes/reducer runs.
    This avoids treating a fixed k-element state vector as time-series data when L happens to equal k."""
    out = FeatureSnapshot()
    codec = codec if codec is not None else Codec()
    out.rowwise = False
    for i, ind in enumerate(strat._indicators):
        name = ind.name if isinstance(ind.name, str) else ",".join(map(str, ind.name))
        key = f"I[{i}]:{name}"
        try:
            out[key] = codec.convert(key, np.array(ind, copy=True))
            out.rowwise |= np.asarray(out[key]).dtype.kind in "biuf"
        except (ValueError, TypeError, OverflowError) as exc:
            out.unassessed[key] = f"categorical capture failed: {exc}"

    def skip(k, v, why, unavailable=True):
        typename = f"{type(v).__module__}.{type(v).__qualname__}"
        out.limitations.append(f"Attribute {k!r} ({typename}) not compared as a separate attribute: {why}.")
        if unavailable:
            # Preserve a named unsupported channel so no-divergence cannot imply complete coverage.
            out.unassessed[f"attr:{k}"] = f"unsupported attribute {typename}: {why}"

    def infer(v):
        if isinstance(v, (pd.Series, pd.DataFrame)):
            return "pandas_rows" if len(v) == n else "state"
        if isinstance(v, (pd.Categorical, pd.Index)):
            return "array_rows" if len(v) == n else "state"
        if isinstance(v, np.ndarray):
            if v.ndim == 0:
                return "scalar"
            if v.ndim == 1:
                return "array_rows" if len(v) == n else "state"
            if v.ndim == 2:
                # NumPy row arrays traditionally use the last axis; sklearn matrices use the first.
                # Keep this decision from the full run when a short prefix becomes square.
                if v.shape[-1] == n:
                    return "array_last_axis"
                if v.shape[0] == n:
                    return "array_first_axis"
            return "state"
        if isinstance(v, (list, tuple)):
            return "state"
        if np.isscalar(v) or v is None or v is pd.NA or v is pd.NaT or isinstance(v, datetime.date):
            return "scalar"
        return "unsupported"

    for k, v in vars(strat).items():
        if k in {"_broker", "_data", "_params", "_indicators"}:
            skip(k, v, "framework container outside feature/state comparison", unavailable=False)
            continue
        if isinstance(v, _Indicator):
            skip(k, v, "Indicator alias already captured in I[...] channels", unavailable=False)
            continue
        try:
            empty_container = isinstance(v, (dict, set, frozenset, list, tuple)) and not v
            mode = layouts.setdefault(k, "empty_container" if empty_container else infer(v))
            if mode == "empty_container":
                skip(k, v, "EXCLUDED empty metadata container" if empty_container else "empty metadata container changed",
                     unavailable=not empty_container)
                continue
            key = f"attr:{k}"
            if mode == "pandas_rows":
                if not isinstance(v, (pd.Series, pd.DataFrame)) or len(v) != n:
                    skip(k, v, "row-aligned pandas layout changed")
                    continue
                if isinstance(v, pd.DataFrame) and (not v.columns.is_unique or len(set(map(str, v.columns))) != len(v.columns)):
                    skip(k, v, "duplicate column names cannot identify separate channels")
                    continue
                if isinstance(v, pd.DataFrame) and v.shape[1] == 0:
                    skip(k, v, "empty row-aligned DataFrame has no numeric coverage")
                    continue
                if v.index.equals(strat.data.index):
                    columns = v.items() if isinstance(v, pd.DataFrame) else [(None, v)]
                    for col, values in columns:
                        channel = f"{key}.{col}" if isinstance(v, pd.DataFrame) else key
                        try:
                            converted = codec.convert(channel, values)
                            out[channel] = converted.copy() if isinstance(converted, CategoricalValues) else values.copy(deep=True)
                            out.rowwise |= np.asarray(converted).dtype.kind in "biuf"
                        except (ValueError, TypeError, OverflowError) as exc:
                            out.unassessed[channel] = f"categorical capture failed: {exc}"
                else:
                    out[key] = v.copy(deep=True)  # core reports exact index mismatch
                if not v.index.equals(strat.data.index):
                    out.limitations.append(f"Attribute {k!r} ({type(v).__module__}.{type(v).__qualname__}) not compared: row index differs from the strategy input.")
            elif mode.startswith("array_"):
                converted = codec.convert(key, v)
                arr = np.asanyarray(converted)
                axis = 0 if mode in ("array_rows", "array_first_axis") else -1
                expected_ndim = 1 if mode == "array_rows" else 2
                if arr.ndim != expected_ndim or arr.shape[axis] != n:
                    skip(k, v, "row-aligned array layout changed")
                    continue
                if not arr.size or arr.dtype.kind not in "biuf":
                    skip(k, v, f"empty or non-numeric row array (dtype {arr.dtype})")
                    continue
                out[key] = (arr.T if mode == "array_first_axis" else arr).copy()
                out.rowwise = True
            elif mode == "scalar":
                arr = np.asanyarray(codec.convert(key, v))
                if arr.ndim != 0:
                    skip(k, v, "scalar layout changed")
                    continue
                if arr.dtype.kind not in "biuf":
                    skip(k, v, f"non-numeric/complex scalar (dtype {arr.dtype})")
                    continue
                out[key] = np.full(n, arr[()], dtype=arr.dtype)
                if isinstance(arr, CategoricalValues):
                    out[key] = out[key].view(CategoricalValues)
            elif mode == "state":
                size = _state_size(v)
                if size > MAX_STATE_ELEMENTS or size * n > MAX_STATE_CELLS:
                    skip(k, v, f"state expansion cap ({MAX_STATE_ELEMENTS} elements, {MAX_STATE_CELLS} broadcast cells); tuple/list row arrays require an ndarray row layout")
                    continue
                try:
                    arr = np.asanyarray(codec.convert(key, v))
                except (ValueError, TypeError) as exc:
                    skip(k, v, f"ragged or unsupported state vector: {str(exc)[:200]}")
                    continue
                shape_key = key + "[shape]"
                if arr.dtype.kind in "biuf" and (arr.size or shape_key in codec.encodings):
                    # Shape is a value of init state too. Missing elements in a
                    # shorter state vector must not merely erase the comparison.
                    code = codec.convert(shape_key, str(arr.shape))
                    out[shape_key] = np.full(n, code[()], dtype=float).view(CategoricalValues)
                if arr.size == 0:
                    skip(k, v, "empty state vector has no numeric coverage")
                    continue
                if arr.dtype.kind not in "biuf":
                    # Never broadcast nested object elements: it can raise, or accidentally
                    # turn a list inside an object array into an apparent row-wise feature.
                    skip(k, v, f"non-numeric/complex state vector (dtype {arr.dtype})")
                    continue
                for ix in np.ndindex(arr.shape):
                    channel = key + "[" + ",".join(map(str, ix)) + "]"
                    out[channel] = np.full(n, arr[ix], dtype=arr.dtype)
                    if isinstance(arr, CategoricalValues):
                        out[channel] = out[channel].view(CategoricalValues)
            else:
                skip(k, v, "unsupported state type; object/dict internals are not inspected")
                continue
        except Exception as exc:
            # Custom array/Series subclasses can fail during conversion or copying.
            # Remove any partial capture of this attribute, then report it by name/type.
            for channel in list(out):
                if channel == f"attr:{k}" or (channel.startswith(f"attr:{k}[") or channel.startswith(f"attr:{k}.")):
                    del out[channel]
            skip(k, v, f"attribute capture failed ({type(exc).__name__}): {str(exc)[:200]}")
    return out


def make_feature_fn(strategy: type, prepare: Optional[Callable[[pd.DataFrame], pd.DataFrame]],
                    params: Dict[str, Any], backtest_kwargs: Dict[str, Any]) -> Callable[[pd.DataFrame], dict]:
    _, Backtest, _Indicator = _backtesting()
    codec = Codec()
    layouts = {}  # classification from the first/full run; fixed across all shorter histories

    def features(rows: pd.DataFrame) -> dict:
        df = prepare(rows.copy(deep=True)) if prepare else rows
        if not isinstance(df, pd.DataFrame) or not df.index.equals(rows.index):
            raise ValueError("prepare() must return a DataFrame with the same index, in the same order, as its input")
        if not df.columns.is_unique or len(set(map(str, df.columns))) != len(df.columns):
            raise ValueError("prepare() returned duplicate column names; channels cannot be identified separately")
        holder: Dict[str, Dict[str, np.ndarray]] = {}
        n = len(df)

        class Snap(strategy):  # snapshot right after init(), then stop before next()
            def init(self):
                super().init()
                holder["s"] = _snapshot(self, n, _Indicator, layouts, codec)
                raise _StopAfterInit

        Snap.__name__ = Snap.__qualname__ = strategy.__name__
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                Backtest(df, Snap, **backtest_kwargs).run(**params)
            except _StopAfterInit:
                pass
        if "s" not in holder:
            raise RuntimeError("Strategy.init() did not complete")
        out = holder["s"]
        if prepare:
            for c in df.columns:
                if c not in rows.columns:
                    key = f"data:{c}"
                    try:
                        out[key] = codec.convert(key, df[c])
                        out.rowwise |= np.asarray(out[key]).dtype.kind in "biuf"
                    except (ValueError, TypeError, OverflowError) as exc:
                        out.unassessed[key] = f"categorical capture failed: {exc}"
        if not out.rowwise:
            out.limitations.append(INIT_STATE_ONLY)
        return out

    # A reducer candidate must infer layout from its own long input, so its claimed
    # reproducer works independently of the original history's inferred shapes.
    features._nullcheck_input_roots = (strategy, prepare)
    features._nullcheck_input_values = params
    features._nullcheck_codec = codec
    features._nullcheck_layouts = layouts
    features._nullcheck_fresh = lambda: make_feature_fn(strategy, prepare, params, backtest_kwargs)
    return features


def check_strategy(strategy: type, data: pd.DataFrame, settings: Settings = Settings(), *,
                   prepare: Optional[Callable] = None, params: Optional[Dict[str, Any]] = None,
                   backtest_kwargs: Optional[Dict[str, Any]] = None) -> Tuple[Result, Dict[str, Any]]:
    """LEAK check for a backtesting.py Strategy on RAW rows. Derived columns belong in prepare(raw_rows)."""
    params, backtest_kwargs = dict(params or {}), dict(backtest_kwargs or {})
    extra = [str(c) for c in data.columns if c not in OHLCV]
    lim = ["Indicators are snapshotted at the end of Strategy.init(); next(), trades, orders and equity are not checked."]
    reason = None
    if extra:
        reason = (f"Input already contains derived column(s) {extra[:10]}; they are sliced, not recomputed from raw rows. "
                  "Build them in prepare().")
        lim.append(reason)
    fn = make_feature_fn(strategy, prepare, params, backtest_kwargs)
    res = check(fn, data, settings, extra_limitations=lim, raw_prefix_recomputed=not extra,
                extra_not_assessed_reason=reason)
    return res, {"params": params, "backtest_kwargs": backtest_kwargs}


def framework_version() -> str:
    return _backtesting()[0].__version__

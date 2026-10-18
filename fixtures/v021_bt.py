# nullcheck test fixture: MIT. Imports unbundled backtesting.py (AGPL-3.0; PyPI classifier: or later); see README "Licences".
"""Disclosed regression repros from an internal v0.2.0 review (B1).
No new held-out evaluation cases are included. Imports optional unbundled backtesting 0.6.6.
"""
import numpy as np
import pandas as pd
from backtesting import Strategy

def prices(n=500, seed=0):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, .01, n)))
    idx = pd.bdate_range("2021-01-04", periods=n, name="Date")
    return pd.DataFrame({"Open": c * (1 + rng.normal(0, .002, n)), "High": c * 1.01, "Low": c * .99, "Close": c,
                         "Volume": rng.integers(1000, 2000, n).astype(float)}, index=idx)

def sma(c, n): return pd.Series(c).rolling(n).mean().to_numpy()

class NdarrayLevels(Strategy):          # same leak as blind-suite M6 but without list(...)
    def init(self):
        self.s = self.I(sma, self.data.Close, 10)
        self.levels = np.percentile(self.data.Close, [10, 50, 90])
    def next(self): pass

class SeriesQuantiles(Strategy):
    def init(self):
        self.s = self.I(sma, self.data.Close, 10)
        self.q = self.data.df.Close.quantile([.1, .9])
    def next(self): pass

class NpBoolScalar(Strategy):
    def init(self):
        self.s = self.I(sma, self.data.Close, 10)
        self.bull = self.data.Close[-1] > self.data.Close[0]     # np.bool_
    def next(self): pass

class FeatureMatrix(Strategy):           # sklearn-style (n_rows, n_features) with full-sample z-score
    def init(self):
        self.s = self.I(sma, self.data.Close, 10)
        c = np.asarray(self.data.Close)
        self.X = np.column_stack([(c - c.mean()) / c.std(), np.r_[np.nan, np.diff(c)]])
    def next(self): pass

class DictState(Strategy):
    def init(self):
        self.s = self.I(sma, self.data.Close, 10)
        self.lv = {"hi": float(np.max(self.data.Close))}
    def next(self): pass

class FloatScalarOK(Strategy):           # control: float() form, as in M6
    def init(self):
        self.s = self.I(sma, self.data.Close, 10)
        self.levels = float(np.percentile(self.data.Close, 90))
    def next(self): pass

class StrListState(Strategy):
    def init(self):
        self.s = self.I(sma, self.data.Close, 10)
        self.feature_names = ["sma10", "ret"]
    def next(self): pass

class ArrayList(Strategy):
    def init(self):
        c = np.asarray(self.data.Close)
        self.bands = [sma(c, 10) * 1.02, sma(c, 10) * .98]
    def next(self): pass

class ConstantStates(Strategy):
    def init(self):
        self.levels = np.array([10., 50., 90.])
        self.q = pd.Series([1., 2.], index=[.1, .9])
        self.bull = np.bool_(True)
        self._state_matrix = np.array([[1., 2.], [3., 4.]])
    def next(self):
        raise AssertionError("next not part of check")


class CausalMatrix(Strategy):
    def init(self):
        c = np.asarray(self.data.Close)
        self.rows = np.column_stack([c, np.r_[np.nan, np.diff(c)]])
        self.last_axis = np.vstack([c, np.r_[np.nan, np.diff(c)]])
    def next(self):
        raise AssertionError("next not part of check")


class UnsupportedState(Strategy):
    def init(self):
        self.causal = np.asarray(self.data.Close).copy()
        self.metadata = {"name": "demo"}
        self.description = "demo"
        self._opaque = object()
    def next(self):
        raise AssertionError("next not part of check")


class QuantileWithIndicator(Strategy):
    """R4b: ndarray quantile state alongside a causal self.I, from the authorized report."""
    def init(self):
        self.s = self.I(sma, self.data.Close, 10)
        self.bounds = np.quantile(self.data.Close, [.1, .9])
    def next(self): pass


class TensorState(Strategy):
    def init(self):
        self.s = self.I(sma, self.data.Close, 10)
        self.tensor = np.full((2, 3, 4), np.asarray(self.data.Close).mean())
    def next(self): pass


class ConstantMultiShape(Strategy):
    def init(self):
        self.zero_dim = np.array(7.)
        self.tensor = np.arange(24.).reshape(2, 3, 4)
        self.pair = (1., 2.)
        self.floating = np.float32(0.25)
        self.integer = np.uint64(42)
    def next(self): pass


class FailingArray(np.ndarray):
    def __new__(cls):
        return np.array([1., 2.]).view(cls)
    @property
    def ndim(self):
        raise ValueError("deliberate subclass capture failure")


class UnsupportedGallery(Strategy):
    """Unsupported state is individually visible even beside supported numeric output."""
    def init(self):
        self.causal = np.asarray(self.data.Close).copy()
        self.mapping = {"threshold": 7.}
        self.timestamp = pd.Timestamp("2020-01-01")
        self.mixed_tuple = (1., pd.Timestamp("2020-01-01"))
        self.numpy_date = np.datetime64("2020-01-01")
        self.numpy_text = np.str_("demo")
        self.numpy_complex = np.complex64(1+2j)
        self.zero_dim_date = np.array(np.datetime64("2020-01-01", "ns"))
        self.structured = np.array([(1., 2)], dtype=[("a", "f8"), ("b", "i8")])
        self.numpy_void = self.structured[0]
        self.nested = np.empty(2, dtype=object)
        self.nested[0] = np.array([1., 2.])
        self.nested[1] = {"threshold": 7.}
        self.ragged_tuple = (np.array([1., 2.]), np.array([3.]))
        self.series_objects = pd.Series([{"a": 1.}, {"b": 2.}])
        self.empty_tensor = np.empty((2, 0, 3))
        self.empty_rows = np.empty((len(self.data), 0))
        self.empty_frame = pd.DataFrame(index=self.data.index)
        self.failing_array = FailingArray()
    def next(self): pass


class UnsupportedWithLeak(UnsupportedGallery):
    def init(self):
        super().init()
        self.levels = np.quantile(self.data.Close, [.1, .9])

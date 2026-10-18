"""Regression strategies from disclosed scalar/list and TrailingStrategy leak classes.
Imports unbundled backtesting 0.6.6, labelled AGPL-3.0 (classifier: or later).
No upstream library source is copied here.
"""
import numpy as np
import pandas as pd
from backtesting import Strategy
from backtesting.lib import SignalStrategy, TrailingStrategy


class ScalarLeak(Strategy):
    def init(self):
        self.thr = float(np.percentile(self.data.Close, 90))
    def next(self):
        raise AssertionError("next must never run")


class ListLeak(Strategy):
    def init(self):
        self.levels = list(np.percentile(self.data.Close, [10, 50, 90]))
    def next(self):
        raise AssertionError("next must never run")


class M6(ScalarLeak):
    def init(self):
        super().init()
        self.levels = list(np.percentile(self.data.Close, [10, 50, 90]))
        self.sma = self.I(lambda c: pd.Series(c).rolling(10).mean(), self.data.Close, name="sma10")


class ConstantState(Strategy):
    def init(self):
        self.threshold = 90.
        self.levels = [10., 50., 90.]
        self._private = np.asarray(self.data.Close).copy()
    def next(self):
        raise AssertionError("next must never run")


class SparseSignal(SignalStrategy):
    def init(self):
        super().init()
        self.set_signal((np.asarray(self.data.Close) > np.asarray(self.data.Open)) * .5)


class TrailingSparse(SignalStrategy, TrailingStrategy):
    def init(self):
        super().init()
        # All-NaN entry_size must not hide the inherited private ATR back-fill.
        self.set_signal(np.zeros(len(self.data)))
        self.set_trailing_sl(2)


class TrailingPct(TrailingStrategy):
    def init(self):
        super().init()
        self.set_trailing_pct(.05)
    def next(self):
        raise AssertionError("next must never run")


class NextOnlyMutation(Strategy):
    def init(self):
        self.trace = np.arange(len(self.data), dtype=float)
    def next(self):
        self.trace[:] = -999
        raise AssertionError("next must never run")

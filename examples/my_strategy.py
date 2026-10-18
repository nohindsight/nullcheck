# nullcheck source: MIT. Imports unbundled backtesting.py, labelled AGPL-3.0
# (upstream PyPI classifier: or later). Redistribution/service combinations
# must satisfy applicable upstream licence obligations; see README Licences.
"""Quickstart backtesting.py strategy. Derived columns are built in prepare() from raw rows, so nullcheck can
recompute them on every truncated history."""
import pandas as pd
from backtesting import Strategy


def prepare(raw):
    out = raw.copy()
    out["Trend"] = out["Close"].rolling(50, center=True).mean()   # centred window: peeks 25 rows ahead
    return out


class MyStrategy(Strategy):
    def init(self):
        self.trend = self.I(lambda: self.data.Trend, name="trend")
        self.sma = self.I(lambda c: pd.Series(c).rolling(10).mean(), self.data.Close, name="sma10")

    def next(self):
        if self.sma[-1] > self.trend[-1] and not self.position:
            self.buy()
        elif self.sma[-1] < self.trend[-1] and self.position:
            self.position.close()

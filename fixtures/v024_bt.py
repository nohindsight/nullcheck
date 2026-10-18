# nullcheck test fixture: MIT. Imports unbundled backtesting.py (AGPL-3.0; PyPI classifier: or later); see README "Licences".
"""Disclosed categorical regression patterns; init snapshots only, no trading."""
import numpy as np
import pandas as pd
from backtesting import Strategy


def sma(c): return pd.Series(c).rolling(5).mean().to_numpy()


class Base(Strategy):
    def init(self): self.s = self.I(sma, self.data.Close)
    def next(self): pass


class RegimeStr(Base):
    def init(self):
        super().init()
        self.regime = "bull" if self.data.Close[-1] > self.data.Close[0] else "bear"


class PeakDate(Base):
    def init(self):
        super().init()
        self.peak = self.data.df.Close.idxmax()


class LabelRows(Base):
    def init(self):
        super().init()
        c = self.data.df.Close
        self.labels = np.where(c.shift(-1) > c, "up", "down")


class LabelCategory(LabelRows):
    def init(self):
        super().init()
        self.labels = pd.Series(self.labels, index=self.data.index, dtype="category")


class FutureDates(Base):
    def init(self):
        super().init()
        c = self.data.df.Close
        self.dates = c.index[(c.shift(-5) > c * 1.02).to_numpy()].values


class SplitDate(Base):
    def init(self):
        super().init()
        self.split = self.data.index[int(len(self.data) * .7)]


class NoneSentinel(Base):
    def init(self):
        super().init()
        self.target = None if self.data.Close[-1] > self.data.Close[0] else float(np.percentile(self.data.Close, 90))


class NumericToNone(Base):
    def init(self):
        super().init()
        self.target = 1. if len(self.data) >= 60 else None


def prepare_label(rows):
    rows["target"] = np.where(rows.Close.shift(-1) > rows.Close, "up", "down")
    return rows


class MixedFrameLeak(Base):
    def init(self):
        super().init()
        self.frame = self.data.df[["Close"]].copy()
        self.frame["label"] = np.where(self.frame.Close.shift(-1) > self.frame.Close, "up", "down")


class CategoricalOnly(Strategy):
    def init(self):
        self.labels = np.where(self.data.df.Close.diff() > 0, "up", "down")
    def next(self): pass


class ConstantMetadata(Base):
    def init(self):
        super().init()
        self.mode = "long_only"
        self.missing = None
        self.date = pd.Timestamp("2020-01-01", tz="UTC")
        self.names = ["close", "volume"]
        self.categories = pd.Categorical(["first", "second"])


class DatetimeRows(Base):
    def init(self):
        super().init()
        self.events = self.data.index.to_numpy().copy()
        self.events[:-1] = self.events[1:]


class StringMatrix(Base):
    def init(self):
        super().init()
        c = self.data.df.Close
        labels = np.where(c.shift(-1) > c, "up", "down")
        self.labels = np.column_stack([labels, labels])


class EncodingSwitch(Strategy):
    def init(self):
        self.feature = (np.sin(np.arange(len(self.data))) if self.data.Close[0] > 100
                        else np.full(len(self.data), "label"))
    def next(self): pass


class DuplicateFrame(Base):
    def init(self):
        super().init()
        c = self.data.df.Close
        self.frame = pd.DataFrame({"leak":np.where(c.shift(-1)>c, "up", "down"), "constant":"flat"}, index=c.index)
        self.frame.columns = ["label", "label"]


class StringifiedDuplicateFrame(DuplicateFrame):
    def init(self):
        super().init()
        self.frame.columns = [1, "1"]


def prepare_duplicate(rows):
    frame = pd.DataFrame({"a":np.where(rows.Close.shift(-1)>rows.Close, "up", "down"), "b":"flat"}, index=rows.index)
    frame.columns = ["label", "label"]
    return pd.concat([rows, frame], axis=1)

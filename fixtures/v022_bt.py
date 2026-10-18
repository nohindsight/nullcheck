# nullcheck test fixture: MIT. Imports unbundled backtesting.py (AGPL-3.0; PyPI classifier: or later); see README "Licences".
"""Synthetic K1/K2 controls; no financial performance interpretation."""
import numpy as np
import pandas as pd
from backtesting import Strategy
from fixtures import v022_features as F

class TupleRows(Strategy):
    def init(self):
        self.bands = (np.asarray(self.data.Close).copy(), np.asarray(self.data.Close).copy())
        self.causal = self.data.df.Close.rolling(2, min_periods=1).mean().to_numpy()
    def next(self): pass

class CategoricalState(Strategy):
    def init(self):
        self.mode = "long_only"
        self.last_signal = None
        self.labels = pd.Categorical(["long", "flat"])
        self.names = ["close", "volume"]
        self.object_strings = pd.Series(["long", None])
        self.object_array = np.array(["long", None], dtype=object)
        self.category_series = pd.Series(["long", "flat"], dtype="category")
        self.causal = self.data.df.Close.rolling(2, min_periods=1).mean().to_numpy()
    def next(self): pass

class CategoricalAndLeak(CategoricalState):
    def init(self):
        super().init()
        self.future = self.data.df.Close.shift(-1)

class ExternalStrategy(Strategy):
    def init(self):
        self.prediction = F.external_file(self.data.df)
    def next(self): pass

class PreparedStrategy(Strategy):
    def init(self):
        self.prediction = np.asarray(self.data.prediction)
    def next(self): pass

class ParameterPredictions(Strategy):
    predictions = None
    def init(self):
        self.signal = np.asarray(self.predictions)[:len(self.data)]
    def next(self): pass


class NumericLookupTable(Strategy):
    def init(self):
        self.causal = self.data.df.Close.rolling(2, min_periods=1).mean().to_numpy()
        self.table = np.arange(1200.).reshape(300, 4)
    def next(self): pass

class DatetimeMetadata(Strategy):
    def init(self):
        self.causal = self.data.df.Close.rolling(2, min_periods=1).mean().to_numpy()
        self.lastday = self.data.index.values[-1]
        self.timestamp = pd.Timestamp(self.data.index[-1])
        self.events = {}
        self.days = set()
    def next(self): pass

class EmptyToPopulated(DatetimeMetadata):
    def init(self):
        super().init()
        self.events = {} if len(self.data) > 20 else {"future": 3.}

class ScalarOnly(Strategy):
    def init(self):
        self.count = 0
        self.label = None
    def next(self): pass

class MixedFrame(Strategy):
    def init(self):
        self.frame = self.data.df[["Close"]].copy()
        self.frame["signal"] = self.frame.Close.rolling(2, min_periods=1).mean()
        self.frame["label"] = "long"
        self.frame["date"] = self.frame.index
    def next(self): pass

class WrongIndexFrame(MixedFrame):
    def init(self):
        super().init()
        self.frame = self.frame.reset_index(drop=True)


def prepare_metadata(rows):
    rows["Local time"] = rows.index.strftime("%d.%m.%Y %H:%M:%S")
    rows["date"] = rows.index
    rows["signal"] = rows.Close.rolling(2).mean()
    return rows

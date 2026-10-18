# nullcheck test fixture: MIT. Imports unbundled backtesting.py (AGPL-3.0; PyPI classifier: or later); see README "Licences".
import numpy as np
import pandas as pd
from backtesting import Strategy

class RowLocal(Strategy):
    def init(self):
        self.hl2 = self.I(lambda h, l: (h+l)/2, self.data.High, self.data.Low)
        self.ret = self.I(lambda c, o: c/o-1, self.data.Close, self.data.Open)
    def next(self): pass

class EmptyList(Strategy):
    def init(self):
        self.sma = self.data.df.Close.rolling(5).mean()
        self.history = []
        self.pending = ()
    def next(self): pass

class EmptyThenPopulated(EmptyList):
    def init(self):
        super().init()
        self.history = [] if len(self.data)>50 else ["future"]

class EmptyOnly(Strategy):
    def init(self): self.history = []
    def next(self): pass

# nullcheck test fixture: MIT. Imports unbundled backtesting.py (AGPL-3.0; PyPI classifier: or later); see README "Licences".
import numpy as np
import pandas as pd
from backtesting import Strategy


class LastDay(Strategy):
    def init(self):
        self.lastday = self.data.index[-1]
        self.I(lambda x: pd.Series(x).rolling(5).mean(), self.data.Close, name='sma')

    def next(self):
        pass


class NumericAndMetadata(LastDay):
    def init(self):
        super().init()
        self.I(lambda x: np.where(np.arange(len(x)) < 3, 0., np.mean(x)),
               self.data.Close, name='zscore')


class SignalAndMetadata(LastDay):
    def init(self):
        super().init()
        self.I(lambda x: np.repeat(len(x) > 40, len(x)), self.data.Close, name='buy_signal')


class StringRows(Strategy):
    def init(self):
        self.labels = np.repeat('long' if len(self.data) > 40 else 'short', len(self.data))

    def next(self):
        pass


class CategoryRows(StringRows):
    def init(self):
        self.labels = pd.Categorical(np.repeat('long' if len(self.data) > 40 else 'short', len(self.data)))


class MissingString(StringRows):
    def init(self):
        self.mode = 'long' if len(self.data) > 40 else None

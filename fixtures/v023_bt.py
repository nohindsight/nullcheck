# nullcheck test fixture: MIT. Imports unbundled backtesting.py (AGPL-3.0; PyPI classifier: or later); see README "Licences".
"""Synthetic analogues of disclosed T01, B04 and S10 coverage traps."""
import numpy as np
from backtesting import Strategy

class RawPredictions(Strategy):
    def init(self):
        self.entry_bar = None
        self.predictions = np.asarray(self.data.Preds).copy()
    def next(self): raise AssertionError("next is outside coverage")

class CountersOnly(Strategy):
    def init(self):
        self.buy_count = 0
        self.sell_count = 0
        self.total_count = 0
        self.stop_loss_price = None
    def next(self): raise AssertionError("next is outside coverage")

class ConstantOnly(Strategy):
    def init(self): self.dollar_amount = 100.
    def next(self): raise AssertionError("next is outside coverage")

class PreparedCopy(Strategy):
    def init(self): self.copy = np.asarray(self.data.prediction).copy()
    def next(self): raise AssertionError("next is outside coverage")

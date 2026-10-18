# nullcheck source: MIT. Imports unbundled backtesting.py, labelled AGPL-3.0
# (upstream PyPI classifier: or later). Redistribution/service combinations
# must satisfy applicable upstream licence obligations; see README Licences.
"""backtesting.py strategies for the validation suite (requires the optional 'backtesting' package)."""
import numpy as np
import pandas as pd
from backtesting import Strategy
from backtesting.lib import crossover


def sma(x, n):
    return pd.Series(x).rolling(n).mean().to_numpy()


class CleanSmaCross(Strategy):
    fast, slow = 10, 40

    def init(self):
        self.f = self.I(sma, self.data.Close, self.fast)
        self.s = self.I(sma, self.data.Close, self.slow)

    def next(self):
        if crossover(self.f, self.s):
            self.position.close(); self.buy()
        elif crossover(self.s, self.f):
            self.position.close()


class LeakShiftIndicator(CleanSmaCross):      # Strategy.I over a shift(-1) series
    def init(self):
        super().init()
        self.peek = self.I(lambda c: pd.Series(c).shift(-1).to_numpy(), self.data.Close, name="next_close")


class LeakAttrZscore(CleanSmaCross):          # whole-sample z kept as a plain attribute, not via Strategy.I
    def init(self):
        super().init()
        c = np.asarray(self.data.Close)
        self.z = (c - c.mean()) / c.std()


class LeakDfRank(CleanSmaCross):              # whole-sample rank computed from self.data.df in init
    def init(self):
        super().init()
        self.r = self.I(lambda: self.data.df.Close.rank(pct=True).to_numpy(), name="rank")


class UsesPreparedZ(CleanSmaCross):           # reads column 'z' that prepare() (or the user) added to the data
    def init(self):
        super().init()
        self.zi = self.I(lambda: np.asarray(self.data.z), name="z")


def prepare_leaky_z(rows):
    out = rows.copy()
    out["z"] = (out["Close"] - out["Close"].mean()) / out["Close"].std()
    return out


def prepare_clean_z(rows):
    out = rows.copy()
    c = out["Close"]
    out["z"] = (c - c.expanding().mean()) / c.expanding().std()
    return out


class NextMutation(Strategy):                 # causal in init; next() overwrites the attribute (review probe)
    def init(self):
        self.trace = np.arange(len(self.data), dtype=float)
        self._n = len(self.data)

    def next(self):
        self.trace = np.full(self._n, float(len(self.data)))


class SeriesAttr(Strategy):                   # causal pandas Series attribute with the input index (valid)
    def init(self):
        self.feature = self.data.df.Close.rolling(5).mean()

    def next(self):
        pass


class MisalignedInit(Strategy):               # Series with reversed labels (review v011 R1 probe)
    def init(self):
        self.feature = pd.Series(np.asarray(self.data.Close), index=self.data.df.index[::-1])

    def next(self):
        raise AssertionError("not used")

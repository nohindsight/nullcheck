"""Disclosed K4/K6/R02 behavior; synthetic, no financial performance claims."""
import numpy as np
import pandas as pd


def log_close(d): return np.log(d.Close)
def intrabar_return(d): return d.Close / d.Open - 1
def typical_price(d): return (d.High + d.Low) / 2
def row_event(d): return np.where(d.Close > d.Open, 1., np.nan)
def price_volume(d): return d.Close * d.Volume

def sma(d): return d.Close.rolling(5).mean()
def ema(d): return d.Close.ewm(span=10).mean()
def change(d): return d.Close.pct_change(fill_method=None)
def lag(d): return d.Close.shift(1)
def cummax(d): return d.Close.cummax()
def rolling_z(d): return (d.Close - d.Close.rolling(20).mean()) / d.Close.rolling(20).std()
def future(d): return d.Close.shift(-1)
def global_z(d): return (d.Close - d.Close.mean()) / d.Close.std()
def mixed(d): return {"local":log_close(d), "rolling":sma(d)}

class InertProgress:
    def update(self, *args, **kwargs): pass

class DataSource:
    def __init__(self): self.values = np.arange(1000.)
    def update(self, *args): pass
    def predict(self): return self.values

class Mutator:
    def update(self, rows): rows["Close"] = np.arange(len(rows))

PROGRESS = InertProgress()
SOURCE = DataSource()
MUTATOR = Mutator()

def progress_only(d):
    PROGRESS.update(1)
    return sma(d)

def object_data(d):
    SOURCE.update(1)
    return {"sma":sma(d), "predictions":SOURCE.predict()[:len(d)]}

def object_attribute(d):
    SOURCE.update(1)
    return {"sma":sma(d), "predictions":SOURCE.values[:len(d)]}

def object_mutation(d):
    MUTATOR.update(d)
    return sma(d)

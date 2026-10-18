"""Write a fictional OHLCV CSV for the quickstart (seeded random walk; not market data). Usage: python make_demo_data.py OUT.csv"""
import sys

import numpy as np
import pandas as pd

rng = np.random.default_rng(2026)
n = 750
close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, n)))
open_ = close * np.exp(rng.normal(0, 0.004, n))
df = pd.DataFrame({"Open": open_, "High": np.maximum(open_, close) * 1.004, "Low": np.minimum(open_, close) * 0.996,
                   "Close": close, "Volume": rng.integers(1_000, 50_000, n).astype(float)},
                  index=pd.bdate_range("2023-01-02", periods=n, name="Date"))
df.to_csv(sys.argv[1] if len(sys.argv) > 1 else "demo.csv")

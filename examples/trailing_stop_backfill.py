"""Back-filled ATR warm-up: a look-ahead pattern, rewritten from scratch as a small MIT example.

The pattern was reported publicly in backtesting.py issue #963 ("ATR introduces look-ahead bias",
https://github.com/kernc/backtesting.py/issues/963). This file contains no backtesting.py code and does not need it.

A trailing stop sits N_ATR average true ranges (ATR) below the best close since entry. An ATR over PERIOD bars is
undefined for the first PERIOD-1 bars. Filling those bars backwards with the first defined value means that bar 0
uses a number computed from bars 0..PERIOD-1: it knows the future.

Usage, from the repository root:
    python examples/trailing_stop_backfill.py                      # bundled fictional data
    python examples/trailing_stop_backfill.py my_prices.csv        # any CSV with Date,Open,High,Low,Close columns
    python -m nullcheck check examples/trailing_stop_backfill.py:atr_backfilled --data examples/data/sample_ohlcv.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import nullcheck

PERIOD = 100   # bars in the ATR window
N_ATR = 3      # stop distance, in ATRs
SAMPLE = Path(__file__).resolve().parent / "data" / "sample_ohlcv.csv"   # fictional prices, not market data


def true_range(raw):
    prev_close = raw["Close"].shift(1)
    spans = pd.concat([raw["High"] - raw["Low"], (raw["High"] - prev_close).abs(), (raw["Low"] - prev_close).abs()],
                      axis=1)
    return spans.max(axis=1)


def atr_backfilled(raw):
    """LEAKS: the warm-up bars receive the first complete ATR, which is computed from later bars."""
    atr = true_range(raw).rolling(PERIOD).mean()
    return pd.DataFrame({"atr": atr.bfill()})


def atr_causal(raw):
    """Same ATR; the warm-up bars stay undefined until PERIOD bars of history exist."""
    return pd.DataFrame({"atr": true_range(raw).rolling(PERIOD).mean()})


def trailing_stop_return(raw, atr):
    """Long whenever flat, entering at the close; exit when the low reaches the stop (at the open if it gaps below).

    The stop starts N_ATR*ATR below the entry close and only moves up. No entry is made on a bar whose ATR is
    undefined. No costs. Returns the compounded return in percent."""
    o, l, c = (raw[k].to_numpy(dtype=float) for k in ("Open", "Low", "Close"))
    atr = np.asarray(atr, dtype=float)
    equity, entry, stop = 1.0, None, None
    for t in range(len(c)):
        if entry is not None:
            if l[t] <= stop:
                equity *= min(o[t], stop) / entry
                entry = None
            else:
                stop = max(stop, c[t] - N_ATR * atr[t])
        if entry is None and np.isfinite(atr[t]):
            entry, stop = c[t], c[t] - N_ATR * atr[t]
    if entry is not None:
        equity *= c[-1] / entry
    return 100 * (equity - 1)


def main(path):
    raw = pd.read_csv(path, index_col=0, parse_dates=True)
    leaky = nullcheck.check(atr_backfilled, raw)
    causal = nullcheck.check(atr_causal, raw)
    print(nullcheck.render(leaky, "atr_backfilled"))
    print(f"atr_causal: STATUS {causal.status}\n")
    r_leaky = trailing_stop_return(raw, atr_backfilled(raw)["atr"])
    r_causal = trailing_stop_return(raw, atr_causal(raw)["atr"])
    print(f"Trailing-stop return on {Path(path).name} ({len(raw)} bars, PERIOD={PERIOD}, N_ATR={N_ATR}):")
    print(f"  back-filled ATR {r_leaky:8.1f}%")
    print(f"  causal ATR      {r_causal:8.1f}%")
    print(f"  difference      {r_leaky - r_causal:8.1f} percentage points")
    print("The back-filled run trades bars 0..%d with a stop it could not have known then." % (PERIOD - 2))
    print("This illustrates the pattern on one dataset; the size and sign of the difference vary with the data.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else SAMPLE)

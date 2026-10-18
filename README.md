# nullcheck

**Find look-ahead leaks by re-running your feature code on truncated history.** By No Hindsight. MIT licence.

A backtest leaks when an indicator value at row *t* depends on rows after *t*. Common causes:
- a z-score that uses the whole file's mean;
- a centred moving average;
- `bfill()` over gaps, or over an indicator's warm-up rows;
- a weekly bar back-filled onto its own days.

An equity curve or trade list cannot reveal this; only your code can. nullcheck calls your feature code again on the first *L* raw rows, for many *L*. If an earlier value changes, it reports `DIVERGENCE_OBSERVED`, with the row, both values and a small reproducer you can re-run.

Educational software, not investment advice. It says nothing about whether a strategy is profitable. A clean result is **not proof of no leak** (see [Limitations](#limitations)).

## 2-minute quickstart

Needs Python ≥ 3.10. Installs numpy, pandas and jsonschema.

```sh
python3 -m venv nc-venv            # Windows: py -m venv nc-venv
source nc-venv/bin/activate        # Windows: nc-venv\Scripts\activate
pip install nullcheck
```

Save this as `features.py`:

```python
"""Each function takes RAW price rows and returns one value per row."""
import numpy as np
import pandas as pd


def leaky(raw):
    """Normalises with the mean and std of the WHOLE file, so early rows 'know' later prices."""
    c = raw["Close"]
    return pd.DataFrame({"zscore": (c - c.mean()) / c.std()})


def fixed(raw):
    """The same idea, using only past rows."""
    c = raw["Close"]
    return pd.DataFrame({"zscore": (c - c.expanding(20).mean()) / c.expanding(20).std()})


if __name__ == "__main__":  # python features.py -> writes demo.csv (fictional prices, not market data)
    rng = np.random.default_rng(1)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 500)))
    pd.DataFrame({"Close": close}, index=pd.bdate_range("2024-01-01", periods=500, name="Date")).to_csv("demo.csv")
```

Then, in the same folder:

```sh
python features.py
python -m nullcheck check features.py:leaky --data demo.csv --out nc-out
python -m nullcheck replay nc-out
python -m nullcheck check features.py:fixed --data demo.csv
```

What you should see (abridged):

```text
STATUS: DIVERGENCE_OBSERVED                                              <- check leaky, exit code 1
  DIVERGENCE  zscore: row 0 differs at prefix length 1 (rows 0..0) (MISSINGNESS: full run 1.6700174104178522, prefix run missing)
  Small reproducer: original rows 0..1 (2 rows); channel 'zscore' differs between its first 1 rows and all of them. Not guaranteed minimal.

 "outcome": "REPRODUCED",                                                <- replay, exit code 0

STATUS: NO_DIVERGENCE_OBSERVED                                           <- check fixed, exit code 0
```

`replay` re-runs the check from the saved, hash-checked record. Run it from the folder that holds both `features.py` and `nc-out`: record paths are relative to that folder (see [Privacy](#privacy-and-the-replay-record)).

From Python:

```python
import nullcheck
res = nullcheck.check(my_features, raw_df, out="nc-out")
print(nullcheck.render(res))
```

## Demo: a back-filled ATR warm-up

[`examples/trailing_stop_backfill.py`](https://github.com/nohindsight/nullcheck/blob/main/examples/trailing_stop_backfill.py) is a small, self-contained example of the pattern reported publicly in [backtesting.py issue #963](https://github.com/kernc/backtesting.py/issues/963): an ATR's undefined warm-up rows are filled backwards, so a trailing stop on bar 0 uses an ATR computed from bars 0–99. The example is written from scratch and does not use backtesting.py. From a clone of this repository:

```sh
python examples/trailing_stop_backfill.py
```

```text
STATUS: DIVERGENCE_OBSERVED
  DIVERGENCE  atr: row 0 differs at prefix length 1 (rows 0..0) (MISSINGNESS: full run 1.786341359830066, prefix run missing)
  ...
atr_causal: STATUS NO_DIVERGENCE_OBSERVED

Trailing-stop return on sample_ohlcv.csv (750 bars, PERIOD=100, N_ATR=3):
  back-filled ATR     57.3%
  causal ATR          44.4%
  difference          12.9 percentage points
```

The data are fictional, and the difference shows only that the two versions disagree; its size and sign vary with the data. Setting a warm-up length does not hide this: nullcheck still audits warm-up rows for values that appear only when later rows exist.

## What it checks

- **Any feature function** `f(raw_rows)` that returns a DataFrame, Series, array or dict of arrays with one value per input row. pandas outputs must keep the input index. Numbers and booleans (as 0/1) are compared; generic string/object outputs are not assessed.
- **backtesting.py 0.6.6 strategies** (optional, `pip install "nullcheck[backtesting]"`). For each run nullcheck builds a fresh `Backtest` and stops right after `Strategy.init()`. It compares that snapshot: every `self.I(...)` indicator; array, Series and DataFrame attributes set in `init`, including private and name-mangled ones; numeric scalars, lists and other non-row state set in `init`; strings, categories and dates by exact equality; and the columns an optional `prepare(raw_rows)` adds. `next()`, orders, trades and equity are **not** checked. Other backtesting versions are refused.

    ```sh
    python -m nullcheck check-bt examples/my_strategy.py:MyStrategy \
        --prepare examples/my_strategy.py:prepare --data examples/data/sample_ohlcv.csv
    ```

- **Raw rows** as downloaded: a CSV or DataFrame whose first column is a unique, increasing time index. Newest-first, shuffled, duplicate or missing indexes are refused as `NOT_ASSESSED`; sort raw rows first. Leave derived columns out: compute them inside your function (or `prepare`) so they can be recomputed on every truncated history.

### How it works

1. **Repeatability:** the full history and every prefix are run twice. Unstable or non-numeric channels become `NOT_ASSESSED` while the others are still compared. nullcheck does not reset globals, caches or random generators.
2. **Prefixes:** about 24 grid/seeded prefix lengths, plus data-aware cutoffs at missing/zero runs, halted prices, large moves and month boundaries, capped at 128. The report states how many were selected.
3. **Comparison**, for every channel and every row before the cutoff: two finite values diverge if `|a−b| > 1e-12 + 1e-9·max(|a|,|b|)`; exactly one finite value is a `MISSINGNESS` divergence; matching NaNs are skipped and never counted as agreement. Warm-up rows are audited too.
4. **Minimum coverage:** a clean result needs at least one compared channel that responds to the input across rows. Constants, copies of raw columns and row counters do not count.
5. **Reducer:** a bounded search shrinks a divergence to a *small* reproducer, not a guaranteed minimal one.

By default nullcheck runs your code at most 380 times. [COVERAGE.md](https://github.com/nohindsight/nullcheck/blob/main/COVERAGE.md) has the full rules.

### Results and exit codes

| Status | Exit | Meaning |
|---|---|---|
| `DIVERGENCE_OBSERVED` | 1 | A value at an earlier row changed when later rows were removed. You decide whether that dependence is intended. |
| `NO_DIVERGENCE_OBSERVED` | 0 | Meaningful coverage and no compared value differed. **This is not proof of no leak.** |
| `NOT_ASSESSED` | 3 | Coverage is incomplete, for example: your code raised on a prefix, output differed between identical runs, output was non-numeric, derived columns were already in the input, or an external file or global was read. |
| `ERROR` | 4 | The runner itself failed. |
| usage/input error | 2 | |

`replay` exits 0 if the outcome is reproduced, 1 if it differs, and 2 if it is refused or unavailable. Exit 0 is not a trading approval.

## Limitations

- **Not proof of no leak.** Only the listed channels, recomputed by your code from truncated raw rows at the sampled prefixes, are compared. Rare leaks at unsampled rows can be missed. Data that are not point-in-time to begin with (revisions, survivorship, back-adjusted prices, other files or network reads) cannot be fixed by truncation.
- **Available, not traded.** A divergence means future information was AVAILABLE to your code in the observed snapshot, not that it was traded. A last-date timestamp used only for printing, a training label or a plotted value can be a real divergence with no effect on orders. Inspect how the named value is used.
- **Init snapshot only.** For backtesting.py, nullcheck stops after `init()`; `next()` and trade use are not checked.
- **Known limits in 0.2.5:**
  - **K4r:** row-local expressions mixed with the index or position (`Close * dayofweek`, `Close * arange(n)`) can count as meaningful coverage although they cannot read future rows.
  - **K5:** precomputed predictions read only in `next()` are missed and can coexist with a clean result.
  - **K8:** the standalone record validator checks structure and consistency, not the coverage conclusion; `replay` re-runs it.
  - **K10:** large tuples of row arrays exceed the state expansion cap and stay `NOT_ASSESSED`; use a row-aligned ndarray.
  - **K11:** lists or object Series mixing numbers with None/strings use exact equality, so tiny floating-point differences can flag (none observed in the reviewed controls).
  - A DataFrame returned with a RangeIndex instead of the raw time index is `NOT_ASSESSED`; positional comparison is not implemented.
- nullcheck is **not a sandbox**: it runs your code with your permissions.

## Privacy and the replay record

`--out` writes a private folder (format: [REPLAY_FORMAT.md](https://github.com/nohindsight/nullcheck/blob/main/REPLAY_FORMAT.md)) with owner-only permissions. It contains a copy of **your** data rows (`input_snapshot.csv`) and names **your** code files by path and hash. Nothing is uploaded; nullcheck's own code sends nothing anywhere. Paths are relative to `--root` (default: the current folder); code outside it makes exact replay `UNAVAILABLE` instead of copying files. `replay` refuses changed files unless `--allow-modified` is given.

Replaying a folder imports and runs the code it references: only replay records and code you trust. **Do not share the folder**; for bug reports, build a separate synthetic example.

## Licences

nullcheck's own files are MIT ([LICENSE](https://github.com/nohindsight/nullcheck/blob/main/LICENSE)). The optional backtesting.py adapter imports an installed copy of backtesting.py and neither bundles nor modifies it; this package contains no backtesting.py source. backtesting.py's licence and metadata say **AGPL-3.0**, while its PyPI classifier says **AGPLv3 or later**. Anyone who redistributes or network-serves nullcheck together with backtesting.py (for example in a Docker image, a frozen executable or a hosted service) must follow the AGPL for that combination, including source availability. nullcheck's own files stay MIT, but the MIT licence grants no rights in backtesting.py. Running both on your own machine is unaffected.

Tested on macOS with Python 3.10 (pandas 2.3, numpy 2.2) and Python 3.12 (pandas 3.0, numpy 2.5), with backtesting 0.6.6; CI runs the tests on Linux. Dependency minimums declare install compatibility, not a tested-version matrix.

## Bugs, questions and contributions

Bug reports are welcome, especially false alarms and missed leaks: please open an issue with a small **synthetic** example (see [CONTRIBUTING.md](https://github.com/nohindsight/nullcheck/blob/main/CONTRIBUTING.md)). Security issues: see [SECURITY.md](https://github.com/nohindsight/nullcheck/blob/main/SECURITY.md). Changes are listed in [CHANGELOG.md](https://github.com/nohindsight/nullcheck/blob/main/CHANGELOG.md).

Most of the code was written by AI agents under the author's direction. Trust it on the strength of its tests and stated limits, and report anything that looks wrong.

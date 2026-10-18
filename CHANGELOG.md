# Changelog

## 0.2.5 — first public release

nullcheck re-runs your feature code on truncated histories and reports values that change when later rows are removed.

- Generic checker for any `raw_rows -> features` function, plus an optional adapter for backtesting.py 0.6.6 that snapshots `Strategy.init()` (indicators, private attributes, scalar/list state, string/category/date metadata, `prepare()` columns).
- Back-fill into warm-up rows is detected: a value present in the full run where the shorter run has NaN, or a different value, is a `MISSINGNESS`/`VALUE` divergence even inside a declared warm-up.
- Automatic prefix planning: a seeded grid plus data-aware cutoffs at gaps, zero runs, halted prices, large moves and month boundaries (capped at 128).
- Refuses newest-first, shuffled, duplicate or missing time indexes instead of guessing.
- Minimum meaningful coverage: constants, copies of raw columns and row counters cannot produce a clean result.
- External files, globals and default-argument state read by your code are named and make their channel `NOT_ASSESSED`.
- Reports list the earliest observed difference for every divergent channel, with readable values; a bounded reducer produces a small reproducer.
- Private, hash-checked replay record (schema 0.4.0) and a `replay` command.
- Known limits K4r, K5, K8, K10 and K11 are documented in the README and COVERAGE.md.

Versions 0.1.0–0.2.4 were private development versions and were not published.

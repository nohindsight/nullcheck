"""nullcheck: find look-ahead leaks by re-running your feature code on truncated history. By No Hindsight, MIT.

    import nullcheck
    res = nullcheck.check(my_features, raw_df, out="nc-out")          # any function raw_rows -> features
    res = nullcheck.check_backtesting(MyStrategy, raw_df, prepare=my_prepare, out="bt-out")
    print(nullcheck.render(res))

Educational software, not investment advice. NO_DIVERGENCE_OBSERVED means only that no difference was observed at
the tested prefixes; it does not prove the code is free of leaks.
"""
__version__ = "0.2.5"

from typing import Any, Callable, Dict, Optional  # noqa: E402

import pandas as pd  # noqa: E402

from .core import Settings, Result, NO_DIVERGENCE, DIVERGENCE, NOT_ASSESSED, ERROR  # noqa: E402,F401
from . import core as _core, replay as _replay  # noqa: E402
from .presentation import earliest_findings  # noqa: E402

EXIT_CODES = {NO_DIVERGENCE: 0, DIVERGENCE: 1, NOT_ASSESSED: 3, ERROR: 4}
STATEMENT = ("NO_DIVERGENCE_OBSERVED is not proof that the code has no leak: only the listed channels, recomputed by "
             "your code from truncated raw rows, at the listed prefixes, were compared. Educational software, not "
             "investment advice; it says nothing about whether a strategy is profitable.")


def _fmt(v):
    return "missing" if v is None else repr(v)


def render(res: Result, title: str = "") -> str:
    L = [f"nullcheck {__version__}  indicator prefix invariance  {title}".rstrip(), "", f"STATUS: {res.status}"]
    if res.reason:
        L.append(f"  reason: {res.reason}")
    earliest = earliest_findings(res)
    n_div = sum(1 for p in res.probes if p["divergence_count"])
    for d in earliest:
        Lp = d["prefix_length"]
        full = d.get("full_display", _fmt(d["full_value"]))
        prefix = d.get("prefix_display", _fmt(d["prefix_value"]))
        L.append(f"  DIVERGENCE  {d['channel']}: row {d['row']} differs at prefix length {Lp} (rows 0..{Lp - 1}) "
                 f"({d['kind']}: full run {full}, prefix run {prefix})")
        if d.get("hint"):
            L.append("    hint: " + d["hint"])
    if res.categorical_channels:
        L.append("  Categorical values use exact equality; numeric tolerances do not apply. Comparison fingerprints are retained in the private record.")
    if earliest:
        L.append("  Headline priority: numeric row indicators/signals before init metadata; actual trade use is not assessed.")
        L.append("  Future information was AVAILABLE in the observed snapshot; this does not establish that it was traded.")
        L.append(f"  {n_div} of {len(res.probes)} probes diverged; rows are the earliest OBSERVED, not necessarily the first leaking row.")
    r = res.reducer
    if r["state"] == "REPRODUCED_SMALLER_CASE":
        rp = r["reproducer"]
        L.append(f"  Small reproducer: original rows {rp['head']}..{rp['stop'] - 1} ({rp['rows']} rows); channel "
                 f"'{rp['channel']}' differs between its first {rp['prefix_stop'] - rp['head']} rows and all of them. Not guaranteed minimal.")
    elif r["state"] != "NOT_ATTEMPTED":
        L.append(f"  Reducer: {r['state']} ({r.get('note', '')})")
    compared = sum(1 for p in res.probes if p["state"] == "COMPARED")
    L += ["", f"Coverage: {len(res.channels)} channel(s), {compared}/{len(res.probes)} prefixes compared, "
              f"{res.runs_used} runs of your code; audited warm-up hint {res.settings.warmup_rows}, terminal exclusion "
              f"{res.settings.terminal_excluded_rows}, tolerance |a-b| > {res.settings.atol:g} + {res.settings.rtol:g}*max(|a|,|b|)."]
    for c in res.channels:
        L.append(f"  channel  {c}")
    for p in res.probes:
        if p["state"] != "COMPARED":
            L.append(f"  {p['state']} prefix {p['prefix_length']}: {p['reason']}")
        elif p["reason"]:
            L.append(f"  PARTIAL prefix {p['prefix_length']}: {p['reason']}")
    for x in sorted(res.limitations, key=lambda item: "framework container" in item or "Indicator alias" in item):
        L.append(f"  limitation: {x}")
    L += ["", STATEMENT]
    return "\n".join(L) + "\n"


def _finish(res: Result, raw: pd.DataFrame, out: Optional[str], target: dict, adapter: dict, framework, title: str,
            original_input: Optional[str] = None, root: Optional[str] = None) -> Result:
    if out:
        _replay.write_record(out, res, raw, target, adapter, framework, render(res, title), original_input, root)
    return res


def check(fn: Callable[[pd.DataFrame], Any], raw: pd.DataFrame, *, out: Optional[str] = None,
          settings: Settings = Settings(), original_input: Optional[str] = None, root: Optional[str] = None) -> Result:
    """Check any function raw_rows -> Series/DataFrame/array/dict of arrays aligned to the rows."""
    res = _core.check(fn, raw, settings)
    return _finish(res, raw, out, {"kind": "callable", "objects": {"fn": fn}},
                   {"id": "python_callback", "version": __version__, "framework_version": None}, None,
                   getattr(fn, "__name__", ""), original_input, root)


def check_backtesting(strategy: type, raw: pd.DataFrame, *, prepare: Optional[Callable] = None,
                      params: Optional[Dict[str, Any]] = None, backtest_kwargs: Optional[Dict[str, Any]] = None,
                      out: Optional[str] = None, settings: Settings = Settings(),
                      original_input: Optional[str] = None, root: Optional[str] = None) -> Result:
    """Check a backtesting.py (0.6.6) Strategy on RAW OHLCV rows; derived columns belong in prepare()."""
    from .adapters import backtesting_py as bp
    res, extra = bp.check_strategy(strategy, raw, settings, prepare=prepare, params=params, backtest_kwargs=backtest_kwargs)
    fv = bp.framework_version()
    return _finish(res, raw, out, {"kind": "backtesting.py", "objects": {"strategy": strategy, "prepare": prepare}, **extra},
                   {"id": "backtesting.py", "version": __version__, "framework_version": fv}, ("backtesting", fv),
                   strategy.__name__, original_input, root)


def replay(folder: str, fn: Optional[Callable] = None, allow_modified: bool = False, root: Optional[str] = None) -> dict:
    return _replay.replay(folder, fn, allow_modified, root)


def validate(folder: str, root: Optional[str] = None) -> list:
    """Validate a record; file paths are resolved against root (default: the current directory)."""
    import os
    return _replay.validate_record(_replay.strict_load(os.path.join(folder, "replay.json")),
                                   os.path.realpath(root or os.getcwd()))

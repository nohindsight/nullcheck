"""CLI.  python -m nullcheck {check,check-bt,replay,validate} ...
check/check-bt exit: 0 no divergence observed at tested prefixes, 1 divergence observed, 2 usage/input error,
3 not assessed (incomplete coverage), 4 error (including a record that failed validation when written).  replay exit: 0 reproduced, 1 different, 2 refused/unavailable/error.
validate exit: 0 valid record, 1 violations."""
import argparse
import json
import sys

import pandas as pd

import nullcheck
from nullcheck.replay import load_by_spec


def _load(path: str, dates: bool) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, float_precision="round_trip")
    if dates:
        df.index = pd.to_datetime(df.index, format="ISO8601")
    return df


def _param(s: str):
    k, _, v = s.partition("=")
    try:
        return k, json.loads(v)
    except json.JSONDecodeError:
        return k, v


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="nullcheck", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("check", "check-bt"):
        s = sub.add_parser(name)
        s.add_argument("target", help="FILE.py:NAME or MODULE:NAME (feature function for check, Strategy class for check-bt)")
        s.add_argument("--data", required=True, help="CSV of RAW rows; first column is the index")
        s.add_argument("--no-dates", action="store_true", help="do not parse the index as dates")
        s.add_argument("--out", help="new folder for the private replay record (inside --root for replay to be available)")
        s.add_argument("--root", help="artifact root that record paths are relative to (default: current directory)")
        s.add_argument("--warmup", type=int, default=0, help="warm-up hint only; warm-up rows are still audited for changed values/missingness")
        s.add_argument("--terminal-exclude", type=int, default=0, help="last rows of each prefix excluded (default 0; >0 hides short-horizon leaks)")
        s.add_argument("--cutoffs", type=int, default=24)
        s.add_argument("--seed", type=int, default=0, help="selects sampled prefix lengths (does not seed your code)")
        s.add_argument("--budget", type=int, default=120, help="callback runs the reducer may use")
        if name == "check-bt":
            s.add_argument("--prepare", help="FILE.py:NAME of prepare(raw_rows) -> DataFrame")
            s.add_argument("--param", action="append", default=[], help="strategy parameter k=v (JSON value)")
            s.add_argument("--bt", action="append", default=[], help="Backtest keyword k=v (JSON value), e.g. cash=10000")
    r = sub.add_parser("replay")
    r.add_argument("folder")
    r.add_argument("--allow-modified", action="store_true", help="replay despite hash mismatches (labelled MODIFIED)")
    r.add_argument("--root", help="artifact root (default: current directory)")
    v = sub.add_parser("validate")
    v.add_argument("folder")
    v.add_argument("--root", help="artifact root (default: current directory)")
    a = p.parse_args(argv)
    try:
        if a.cmd == "replay":
            try:
                res = nullcheck.replay(a.folder, allow_modified=a.allow_modified, root=a.root)
            except Exception as e:  # anything raised while loading or running the recorded code
                print(json.dumps({"outcome": "ERROR", "error": f"{type(e).__name__}: {str(e)[:500]}"}, indent=1))
                return 2
            print(json.dumps(res, indent=1))
            return {"REPRODUCED": 0, "DIFFERENT": 1}.get(res["outcome"].replace("MODIFIED_", ""), 2)
        if a.cmd == "validate":
            try:
                errs = nullcheck.validate(a.folder, a.root)
            except ValueError as e:
                errs = [f"parse: {e}"]
            print("\n".join(errs) if errs else "valid")
            return 1 if errs else 0
        raw = _load(a.data, not a.no_dates)
        target = load_by_spec(a.target)
        st = nullcheck.Settings(warmup_rows=a.warmup, terminal_excluded_rows=a.terminal_exclude, n_cutoffs=a.cutoffs,
                                seed=a.seed, reduce_budget=a.budget)
        if a.cmd == "check":
            res = nullcheck.check(target, raw, out=a.out, settings=st, original_input=a.data, root=a.root)
        else:
            prep = load_by_spec(a.prepare) if a.prepare else None
            res = nullcheck.check_backtesting(target, raw, prepare=prep, params=dict(map(_param, a.param)),
                                              backtest_kwargs=dict(map(_param, a.bt)), out=a.out, settings=st,
                                              original_input=a.data, root=a.root)
        print(nullcheck.render(res, a.target), end="")
        if a.out:
            print(f"Private replay record: {a.out}")
        return nullcheck.EXIT_CODES[res.status]
    except nullcheck._replay.EmissionError as e:
        print(f"nullcheck: error: {e}", file=sys.stderr)
        return 4
    except (OSError, ValueError, LookupError, ImportError, RuntimeError, AttributeError) as e:
        print(f"nullcheck: error: {type(e).__name__}: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

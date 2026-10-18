"""Validation suite: detection and false-flag counts on synthetic fixtures. Usage: python validate.py OUT.json
Population: fictional random-walk OHLCV (seeded), planted leaks vs causal controls written for this suite.
These are counts on THIS population, not a general detection rate."""
import json
import os
import sys
import tempfile
import time

import nullcheck
from fixtures import make_ohlcv
from fixtures.features import LEAKS, CLEAN, WARMUP, unseeded_noise
from nullcheck import Settings
from nullcheck.core import check as core_check

try:
    from fixtures import bt_strategies as B
except ImportError:
    B = None


def run_callable(seeds, n, settings):
    """settings: n_cutoffs, nan_cutoffs, warmup ('declared' uses each control's declared warm-up, else 0), replay."""
    import shutil
    from nullcheck.replay import load_by_spec
    out = {}
    for name, f in list(LEAKS.items()) + list(CLEAN.items()) + [("unseeded_noise", unseeded_noise)]:
        res = {k: 0 for k in ("DIVERGENCE_OBSERVED", "NO_DIVERGENCE_OBSERVED", "NOT_ASSESSED", "ERROR")}
        repro_rows, replay_ok, kinds = [], 0, {}
        w = WARMUP.get(name, 0) if settings["warmup"] == "declared" else 0
        st = Settings(n_cutoffs=settings["n_cutoffs"], nan_cutoffs=settings["nan_cutoffs"], warmup_rows=w)
        for s in seeds:
            df = make_ohlcv(n, s, gaps=6)
            if settings.get("replay"):
                with tempfile.TemporaryDirectory() as t:
                    os.makedirs(os.path.join(t, "code"))
                    shutil.copy(os.path.join(os.path.dirname(__file__), "fixtures", "features.py"), os.path.join(t, "code"))
                    g = load_by_spec(os.path.join(t, "code", "features.py") + ":" + name)
                    r = nullcheck.check(g, df, settings=st, out=os.path.join(t, "r"), root=t)
                    rr = nullcheck.replay(os.path.join(t, "r"), root=t)
                    replay_ok += rr["outcome"] == "REPRODUCED" and rr.get("small_reproducer", "REPRODUCED") == "REPRODUCED"
                    assert not nullcheck.validate(os.path.join(t, "r"), t)
            else:
                r = core_check(f, df, st)
            res[r.status] += 1
            for d in r.divergences:
                kinds[d["kind"]] = kinds.get(d["kind"], 0) + 1
            if r.reducer.get("reproducer"):
                repro_rows.append(r.reducer["reproducer"]["rows"])
        res["kind"] = "leak" if name in LEAKS else ("clean" if name in CLEAN else "nondeterministic")
        res["warmup_rows"] = w
        res["first_divergence_kinds"] = kinds
        if repro_rows:
            res["reproducer_rows_median"] = sorted(repro_rows)[len(repro_rows) // 2]
            res["reproducer_rows_max"] = max(repro_rows)
        if settings.get("replay"):
            res["replays_reproduced"] = replay_ok
        out[name] = res
    return out


def summarise(block):
    def tot(kind, key=None):
        vs = [v for v in block.values() if v["kind"] == kind]
        return sum(v[key] for v in vs) if key else sum(v[k] for v in vs for k in ("DIVERGENCE_OBSERVED", "NO_DIVERGENCE_OBSERVED", "NOT_ASSESSED", "ERROR"))
    return {"leak_runs": tot("leak"), "leak_runs_divergence": tot("leak", "DIVERGENCE_OBSERVED"),
            "clean_runs": tot("clean"), "clean_runs_divergence": tot("clean", "DIVERGENCE_OBSERVED"),
            "clean_runs_no_divergence": tot("clean", "NO_DIVERGENCE_OBSERVED"),
            "clean_runs_not_assessed": tot("clean", "NOT_ASSESSED")}


def main(path):
    t0 = time.time()
    seeds, n = list(range(30)), 500
    report = {"population": f"{len(seeds)} seeded fictional OHLCV series x {n} business days, 6 missing-volume runs each",
              "caveat": "Fixtures and runner were written by the same author; counts describe this population only.",
              "nullcheck": nullcheck.__version__,
              "minimum_coverage_note": "clean_seeded_noise is a causal control but independent of raw values; v0.2.4 expects NOT_ASSESSED (30/240 clean runs), never a leak finding.", "callable": {}}
    for label, st in [("cutoffs24_nan_on_declared_warmup_with_replay", {"n_cutoffs": 24, "nan_cutoffs": True, "warmup": "declared", "replay": True}),
                      ("cutoffs24_nan_on_no_warmup", {"n_cutoffs": 24, "nan_cutoffs": True, "warmup": 0}),
                      ("cutoffs24_nan_off_declared_warmup", {"n_cutoffs": 24, "nan_cutoffs": False, "warmup": "declared"}),
                      ("cutoffs8_nan_off_declared_warmup", {"n_cutoffs": 8, "nan_cutoffs": False, "warmup": "declared"}),
                      ("cutoffs2_nan_off_declared_warmup", {"n_cutoffs": 2, "nan_cutoffs": False, "warmup": "declared"})]:
        blk = run_callable(seeds, n, st)
        report["callable"][label] = {"settings": st, "summary": summarise(blk), "per_fixture": blk}
    if B is not None:
        import backtesting
        cases = [("CleanSmaCross", B.CleanSmaCross, None, {}, "clean"),
                 ("CleanSmaCross+finalize_trades", B.CleanSmaCross, None, {"finalize_trades": True}, "clean"),
                 ("UsesPreparedZ+prepare_clean_z", B.UsesPreparedZ, B.prepare_clean_z, {}, "clean"),
                 ("LeakShiftIndicator", B.LeakShiftIndicator, None, {}, "leak"),
                 ("LeakAttrZscore", B.LeakAttrZscore, None, {}, "leak"),
                 ("LeakDfRank", B.LeakDfRank, None, {}, "leak"),
                 ("UsesPreparedZ+prepare_leaky_z", B.UsesPreparedZ, B.prepare_leaky_z, {}, "leak")]
        bt = {}
        for name, S_, prep, kw, kind in cases:
            c = {"kind": kind, "DIVERGENCE_OBSERVED": 0, "NO_DIVERGENCE_OBSERVED": 0, "NOT_ASSESSED": 0, "ERROR": 0}
            for s in range(10):
                r = nullcheck.check_backtesting(S_, make_ohlcv(400, s), prepare=prep, backtest_kwargs=kw,
                                                settings=Settings(warmup_rows=40))
                c[r.status] += 1
            bt[name] = c
        pre = {}
        for s in range(10):
            df = B.prepare_clean_z(make_ohlcv(400, s))
            r = nullcheck.check_backtesting(B.UsesPreparedZ, df, settings=Settings(warmup_rows=40))
            pre[r.status] = pre.get(r.status, 0) + 1
        bt["UsesPreparedZ, causal z PRECOMPUTED in input (must be NOT_ASSESSED, never clean)"] = pre
        nm = {}
        for s in range(10):
            r = core_check(__import__("nullcheck.adapters.backtesting_py", fromlist=["x"]).make_feature_fn(B.NextMutation, None, {}, {}),
                           make_ohlcv(100, s), Settings(n_cutoffs=6))
            nm[r.status] = nm.get(r.status, 0) + 1
        bt["NextMutation (v0.2 captures _n=len(data) init-state dependence; trace stays causal, next() not run)"] = nm
        report["backtesting_py"] = {"version": backtesting.__version__, "seeds": 10, "rows": 400, "cases": bt}
    report["seconds"] = round(time.time() - t0, 1)
    with open(path, "w") as f:
        json.dump(report, f, indent=1)
    for k, v in report["callable"].items():
        print(k, v["summary"])
    print(json.dumps(report.get("backtesting_py", {}), indent=1))


if __name__ == "__main__":
    main(sys.argv[1])

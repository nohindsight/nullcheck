import contextlib
import io
import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

import nullcheck
from nullcheck import core
from nullcheck.__main__ import main
from fixtures import make_ohlcv
from fixtures.features import LEAKS, CLEAN, WARMUP, unseeded_noise

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
S = nullcheck.Settings

try:
    import backtesting  # noqa: F401
    from fixtures import bt_strategies as B
    HAVE_BT = True
except ImportError:
    HAVE_BT = False


def in_root(t, module, name):
    """Copy a fixture module into root t (as the test's explicit choice) and load `name` from the copy."""
    import shutil
    from nullcheck.replay import load_by_spec
    dst = os.path.join(t, "code", module + ".py")
    if not os.path.exists(dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(os.path.join(ROOT, "fixtures", module + ".py"), dst)
    return load_by_spec(f"{dst}:{name}")


def _load_json(path):
    with open(path) as f:
        return json.load(f)


class TestCore(unittest.TestCase):
    df = make_ohlcv(400, 3, gaps=8)

    def test_every_planted_leak_diverges(self):
        for name, f in LEAKS.items():
            with self.subTest(name):
                self.assertEqual(nullcheck.check(f, self.df).status, nullcheck.DIVERGENCE)

    def test_causal_controls_with_declared_warmup(self):
        for name, f in CLEAN.items():
            with self.subTest(name):
                r = nullcheck.check(f, self.df, settings=S(warmup_rows=WARMUP[name]))
                expected = nullcheck.NOT_ASSESSED if name == "clean_seeded_noise" else nullcheck.NO_DIVERGENCE
                self.assertEqual(r.status, expected, (r.reason, r.divergences[:2]))
                if name == "clean_seeded_noise": self.assertEqual(r.reason, "insufficient meaningful coverage")
                self.assertEqual(r.repeatability, "CONFIRMED")

    def test_undeclared_warmup_is_automatically_skipped_never_divergent(self):
        r = nullcheck.check(CLEAN["clean_long_warmup"], self.df)
        self.assertEqual(r.status, nullcheck.NO_DIVERGENCE)
        self.assertEqual(r.divergences, [])

    def test_missingness_mismatch_is_a_divergence(self):
        r = nullcheck.check(lambda d: d["Close"].shift(-1), self.df)  # noqa: E731
        self.assertEqual(r.status, nullcheck.DIVERGENCE)
        kinds = {d["kind"] for d in r.divergences}
        self.assertEqual(kinds, {"MISSINGNESS"})

    def test_symmetric_tolerance(self):
        # |a-b| = 1.5e-9; threshold atol + rtol*max(|a|,|b|) = 1e-12 + 1e-9*1.0000000015 -> divergence
        self.assertEqual(len(core.compare_channel("x", np.array([1.0]), np.array([1.0 + 1.5e-9]), 1, S())["diffs"]), 1)
        self.assertEqual(len(core.compare_channel("x", np.array([1.0]), np.array([1.0 + 0.5e-9]), 1, S())["diffs"]), 0)

    def test_ranges_partition_prefix(self):
        full = np.array([np.nan, np.nan, 1.0, 2.0, np.inf, 4.0])
        pref = np.array([np.nan, np.nan, 1.0, 2.5, np.inf])
        r = core.compare_channel("x", full, pref, 5, S(warmup_rows=1))
        spans = sorted((x["start"], x["stop"]) for x in r["compared"] + r["skipped"])
        self.assertEqual(spans, [(0, 2), (2, 4), (4, 5)])        # matching warm-up NaN, compared, invalid inf
        self.assertEqual([(d["row"], d["kind"]) for d in r["diffs"]], [(3, "VALUE")])
        self.assertIn("row 4", r["unassessable"])                 # inf pair is invalid, but does not erase row 3 divergence
        r = core.compare_channel("x", full[:4], pref[:4], 4, S())
        self.assertIsNone(r["unassessable"])
        self.assertEqual(r["assessed"], 2)

    def test_tolerance_absorbs_float_noise_only(self):
        f = lambda d: d["Close"].rolling(3).mean() + (1e-13 if len(d) % 2 else 0.0)  # noqa: E731
        self.assertEqual(nullcheck.check(f, self.df, settings=S(warmup_rows=3)).status, nullcheck.NO_DIVERGENCE)
        g = lambda d: d["Close"].rolling(3).mean() + (1e-6 if len(d) % 2 else 0.0)  # noqa: E731
        self.assertEqual(nullcheck.check(g, self.df, settings=S(warmup_rows=3)).status, nullcheck.DIVERGENCE)

    def test_booleans_encoded_strings_not_assessed(self):
        g = lambda d: {"up": (d["Close"].diff() > 0).to_numpy()}  # noqa: E731
        self.assertEqual(nullcheck.check(g, self.df, settings=S(warmup_rows=1)).status, nullcheck.NO_DIVERGENCE)
        f = lambda d: {"lab": np.where(d["Close"] > 100, "hi", "lo")}  # noqa: E731
        r = nullcheck.check(f, self.df)
        self.assertEqual(r.status, nullcheck.NOT_ASSESSED)
        self.assertIn("non-numeric", r.reason)

    def test_misaligned_pandas_output_not_assessed(self):
        r = nullcheck.check(lambda d: d["Close"].reset_index(drop=True), self.df)  # noqa: E731
        self.assertEqual(r.status, nullcheck.NOT_ASSESSED)
        r = nullcheck.check(lambda d: d["Close"].iloc[::-1], self.df)  # noqa: E731
        self.assertEqual(r.status, nullcheck.NOT_ASSESSED)

    def test_settings_validated(self):
        for bad in (S(rtol=-1), S(atol=float("nan")), S(warmup_rows=-1), S(n_cutoffs=0), S(seed=True)):
            with self.assertRaises(ValueError):
                nullcheck.check(CLEAN["clean_drawdown"], self.df, settings=bad)

    def test_probe_cap_and_order(self):
        df = make_ohlcv(400, 3, gaps=300)
        cuts = core.choose_cutoffs(len(df), S(n_cutoffs=200), df)
        self.assertLessEqual(len(cuts), core.MAX_PROBES)
        self.assertEqual(cuts, sorted(set(cuts)))

    def test_reducer_verified_and_within_budget(self):
        r = nullcheck.check(LEAKS["leak_centred_mean"], self.df)
        self.assertEqual(r.reducer["state"], "REPRODUCED_SMALLER_CASE")
        self.assertLessEqual(r.reducer["callback_runs"], r.settings.reduce_budget)
        self.assertLessEqual(r.runs_used, r.max_runs)
        rp = r.reducer["reproducer"]
        self.assertLess(rp["rows"], len(self.df) // 4)
        tiny = nullcheck.check(LEAKS["leak_centred_mean"], self.df, settings=S(reduce_budget=3))
        self.assertEqual(tiny.reducer["state"], "ATTEMPTED_NO_REDUCTION")
        self.assertIsNone(tiny.reducer["reproducer"])


class TestAdversarialProbes(unittest.TestCase):
    """Seven adversarial cases from an internal v0.1.0 review (synthetic 100 rows)."""
    raw = pd.DataFrame({"Open": np.arange(100.) + 100, "High": np.arange(100.) + 102, "Low": np.arange(100.) + 99,
                        "Close": np.arange(100.) + 101, "Volume": np.ones(100)},
                       index=pd.date_range("2020-01-01", periods=100))
    st = S(n_cutoffs=6, reduce_budget=8)

    def status(self, fn, **kw):
        return core.check(fn, self.raw, S(n_cutoffs=6, reduce_budget=8, **kw))

    def test_all_nan_and_all_inf_not_assessed(self):
        for v in (np.nan, np.inf):
            r = self.status(lambda d, v=v: np.full(len(d), v))
            self.assertEqual(r.status, nullcheck.NOT_ASSESSED)

    def test_missing_prefix_features_not_assessed(self):
        n = len(self.raw)
        r = self.status(lambda d: {"x": d.Close.to_numpy()} if len(d) == n else {})
        self.assertEqual(r.status, nullcheck.NOT_ASSESSED)

    def test_some_prefixes_raise_not_assessed(self):
        def f(d):
            if len(d) < 50:
                raise ValueError("short history unsupported")
            return d.Close
        r = self.status(f)
        self.assertEqual(r.status, nullcheck.NOT_ASSESSED)
        self.assertTrue(any("short history" in (p["reason"] or "") for p in r.probes))

    def test_declared_warmup_still_audited_for_divergence(self):
        def f(d):
            a = d.Close.to_numpy().copy()
            a[0] = len(d)
            return a
        self.assertEqual(self.status(f, warmup_rows=10).status, nullcheck.DIVERGENCE)
        self.assertEqual(self.status(f).status, nullcheck.DIVERGENCE)

    def test_prefix_nondeterminism_not_called_a_leak(self):
        rng = np.random.default_rng(19)
        n = len(self.raw)
        r = self.status(lambda d: d.Close.to_numpy() if len(d) == n else rng.normal(size=len(d)))
        self.assertEqual(r.status, nullcheck.NOT_ASSESSED)
        self.assertTrue(all("not repeatable" in p["reason"] for p in r.probes))

    @unittest.skipUnless(HAVE_BT, "backtesting not installed")
    def test_adapter_ignores_next_mutations(self):
        from nullcheck.adapters.backtesting_py import make_feature_fn
        r = core.check(make_feature_fn(B.NextMutation, None, {}, {}), self.raw, self.st)
        # v0.2 now captures _n=len(data), which is real init-state prefix dependence.
        self.assertEqual(r.status, nullcheck.DIVERGENCE)
        self.assertEqual({d["channel"] for d in r.divergences}, {"attr:_n"})
        snap = make_feature_fn(B.NextMutation, None, {}, {})(self.raw)
        np.testing.assert_array_equal(snap["attr:trace"], np.arange(len(self.raw)))


class TestReviewProbes(unittest.TestCase):
    """Additional cases from an internal v0.1.0 review (P1, P2, P5)."""
    raw = TestAdversarialProbes.raw

    def test_second_full_output_missing_not_clean(self):
        calls = [0]

        def f(d):
            calls[0] += 1
            return {} if calls[0] == 2 else {"x": d.Close.to_numpy()}
        r = core.check(f, self.raw, S(n_cutoffs=6))
        self.assertEqual(r.status, nullcheck.NOT_ASSESSED)

    def test_reversed_index_and_infinite_tolerance_and_complex(self):
        r = core.check(lambda d: pd.Series(d.Close.to_numpy(), index=d.index[::-1]), self.raw, S(n_cutoffs=6))
        self.assertEqual(r.status, nullcheck.NOT_ASSESSED)
        with self.assertRaises(ValueError):
            core.check(lambda d: d.Close, self.raw, S(rtol=float("inf")))
        r = core.check(lambda d: np.ones(len(d), dtype=complex) * (1 + len(d) * 1j), self.raw, S(n_cutoffs=6))
        self.assertEqual(r.status, nullcheck.NOT_ASSESSED)
        self.assertIn("complex", r.reason)

    def test_string_NA_input_round_trip_marks_replay_unavailable(self):
        df = self.raw.copy()
        df["Tag"] = "NA"
        with tempfile.TemporaryDirectory() as t:
            out = os.path.join(t, "rec")
            nullcheck.check(in_root(t, "features", "leak_next_bar"), df, out=out, root=t)
            rec = _load_json(os.path.join(out, "replay.json"))
            self.assertEqual(rec["reproduction"]["availability"], "UNAVAILABLE")
            self.assertIn("round-trip", rec["reproduction"]["reason"])
            with tempfile.TemporaryDirectory() as t2:
                out2 = os.path.join(t2, "rec")
                nullcheck.check(in_root(t2, "features", "leak_next_bar"), self.raw, out=out2, root=t2)
                self.assertEqual(_load_json(os.path.join(out2, "replay.json"))["reproduction"]["availability"], "AVAILABLE")

    def test_changed_source_refused_before_execution(self):
        with tempfile.TemporaryDirectory() as t:
            code = os.path.join(t, "owned.py")
            marker = os.path.join(t, "marker")
            src = "import numpy as np\nHELPER = 1\ndef feature(d):\n    return np.full(len(d), float(len(d) * HELPER))\n"
            with open(code, "w") as f:
                f.write(src)
            from nullcheck.replay import load_by_spec
            out = os.path.join(t, "rec")
            nullcheck.check(load_by_spec(code + ":feature"), self.raw, out=out, root=t, settings=S(n_cutoffs=6))
            with open(code, "w") as f:
                f.write(src.replace("HELPER = 1", "HELPER = 22") + f"\nopen({marker!r}, 'w').write('x')\n")
            res = nullcheck.replay(out, root=t)
            self.assertEqual(res["outcome"], "REFUSED")
            self.assertFalse(os.path.exists(marker))


class TestReviewV011(unittest.TestCase):
    """Residuals R1-R4 from an internal v0.1.1 review."""
    raw = TestAdversarialProbes.raw

    def _record(self, t, source, budget=0, **kw):
        path = os.path.join(t, "owned.py")
        with open(path, "w") as f:
            f.write(source)
        from nullcheck.replay import load_by_spec
        out = os.path.join(t, "record")
        nullcheck.check(load_by_spec(path + ":feature"), self.raw, out=out, root=t,
                        settings=S(n_cutoffs=6, reduce_budget=budget, **kw))
        return out, _load_json(os.path.join(out, "replay.json"))

    def _rewrite(self, out, rec):
        p = os.path.join(out, "replay.json")
        os.chmod(p, 0o600)
        with open(p, "w") as f:
            json.dump(rec, f)

    @unittest.skipUnless(HAVE_BT, "backtesting not installed")
    def test_R1_adapter_series_index_checked(self):
        r = nullcheck.check_backtesting(B.MisalignedInit, self.raw, settings=S(n_cutoffs=4, reduce_budget=0))
        self.assertEqual(r.status, nullcheck.NOT_ASSESSED)
        self.assertIn("index", r.reason + " ".join(p["reason"] or "" for p in r.probes))
        r = nullcheck.check_backtesting(B.SeriesAttr, self.raw, settings=S(n_cutoffs=4, warmup_rows=5))
        # Linear input makes this rolling mean indistinguishable from an affine counter.
        self.assertEqual(r.status, nullcheck.NOT_ASSESSED)
        self.assertEqual(r.reason, "insufficient meaningful coverage")
        self.assertIn("attr:feature", r.channels)

    def test_R2_contract_config_mismatch_refused(self):
        with tempfile.TemporaryDirectory() as t:
            out, rec = self._record(t, "import numpy as np\ndef feature(d):\n    return np.ones(len(d)) + (1e-13 if len(d) % 2 else 0.0)\n")
            self.assertEqual(rec["status"], "NOT_ASSESSED")  # constant agreement is insufficient
            rec["contract"]["atol"] = 0.0; rec["contract"]["rtol"] = 0.0
            self._rewrite(out, rec)
            errs = nullcheck.validate(out, t)
            self.assertTrue(any(e.startswith("consistency:") and "atol" in e for e in errs), errs)
            self.assertEqual(nullcheck.replay(out, root=t)["outcome"], "REFUSED")
            self.assertEqual(nullcheck.replay(out, root=t, allow_modified=True)["outcome"], "REFUSED")

    def test_R2_coverage_range_misrepresentation_not_reproduced(self):
        with tempfile.TemporaryDirectory() as t:
            out, rec = self._record(t, "def feature(d):\n    return d.Close\n")
            p = rec["probes"][-1]
            L = p["prefix_length"]
            p["compared_ranges"] = [{"channel": "feature", "start": 3, "stop": L}]
            p["skipped_ranges"] = [{"channel": "feature", "start": 0, "stop": 3, "reason": core.R_INFERRED}]
            self._rewrite(out, rec)                       # structurally and semantically plausible, but untrue
            res = nullcheck.replay(out, root=t)
            self.assertEqual(res["outcome"], "DIFFERENT", res)
            self.assertIn("probes", res["differing_fields"])

    def test_R3_modified_mode_cannot_escape_root(self):
        with tempfile.TemporaryDirectory() as parent:
            t = os.path.join(parent, "root"); os.makedirs(t)
            marker = os.path.join(parent, "marker")
            outside = os.path.join(parent, "outside.py")
            with open(outside, "w") as f:
                f.write(f"open({marker!r}, 'w').write('x')\ndef feature(d):\n    return d.Close\n")
            out, rec = self._record(t, "def feature(d):\n    return d.Close\n")
            h = nullcheck._replay.sha256_file(outside)
            for variant in ("traversal", "symlink"):
                r2 = json.loads(json.dumps(rec))
                for f in r2["files"]:
                    if f["role"] == "SOURCE":
                        if variant == "traversal":
                            f["path"] = "../outside.py"
                        else:
                            link = os.path.join(t, "link.py")
                            if not os.path.exists(link):
                                os.symlink(outside, link)
                            f["path"] = "link.py"
                        f["sha256"] = h
                self._rewrite(out, r2)
                for mod in (False, True):
                    res = nullcheck.replay(out, root=t, allow_modified=mod)
                    self.assertEqual(res["outcome"], "REFUSED", (variant, mod, res))
                    self.assertFalse(os.path.exists(marker), (variant, mod))
            self._rewrite(out, rec)
            with open(os.path.join(t, "owned.py"), "a") as f:   # in-root trusted change: modified mode only
                f.write("# edit\n")
            self.assertEqual(nullcheck.replay(out, root=t)["outcome"], "REFUSED")
            self.assertEqual(nullcheck.replay(out, root=t, allow_modified=True)["outcome"], "MODIFIED_REPRODUCED")

    def test_R4_missing_jsonschema_fails_closed(self):
        import builtins
        with tempfile.TemporaryDirectory() as t:
            out, rec = self._record(t, "def feature(d):\n    return d.Close\n")
        real = builtins.__import__

        def no_js(name, *a, **kw):
            if name == "jsonschema":
                raise ImportError("simulated")
            return real(name, *a, **kw)
        builtins.__import__ = no_js
        try:
            errs = nullcheck._replay.validate_record(rec)
        finally:
            builtins.__import__ = real
        self.assertTrue(errs and errs[0].startswith("schema:"))

    def test_R4_vacuous_compared_probes_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            out, rec = self._record(t, "def feature(d):\n    return d.Close\n")
        bad = json.loads(json.dumps(rec))
        for p in bad["probes"][:-1]:
            p["compared_ranges"] = []
            p["skipped_ranges"] = [{"channel": c, "start": 0, "stop": p["prefix_length"], "reason": "Arbitrary omission"}
                                   for c in bad["coverage"]["channels"]]
        errs = nullcheck._replay.validate_record(bad)
        self.assertTrue(any("COMPARED probe without" in e for e in errs), errs)
        self.assertTrue(any("skipped ranges other than" in e for e in errs), errs)
        bad = json.loads(json.dumps(rec)); bad["contract"]["rtol"] = True
        self.assertTrue(nullcheck._replay.validate_record(bad))

    def test_R4_overflow_number_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            p = os.path.join(t, "x.json")
            with open(p, "w") as f:
                f.write('{"x": 1e999}')
            with self.assertRaises(ValueError):
                nullcheck._replay.strict_load(p)
            with open(p, "w") as f:
                f.write('{"x": -1e999}')
            with self.assertRaises(ValueError):
                nullcheck._replay.strict_load(p)

    def test_R4_malformed_reproducer_rejected_before_import(self):
        with tempfile.TemporaryDirectory() as t:
            marker = os.path.join(t, "marker")
            src = ("import numpy as np\nimport os\nif os.environ.get('NC_MARK'):\n    open(os.environ['NC_MARK'], 'w').write('x')\n"
                   "def feature(d):\n    return (d.Close - d.Close.mean())\n")
            out, rec = self._record(t, src, budget=120)
            self.assertEqual(rec["status"], "DIVERGENCE_OBSERVED")
            fid = rec["reproduction"]["reducer"]["reproducer_file_id"]
            self.assertTrue(fid)
            f = next(x for x in rec["files"] if x["id"] == fid)
            path = os.path.join(t, f["path"])
            os.chmod(path, 0o600)
            with open(path) as fh:
                body = fh.read()
            with open(path, "w") as fh:
                fh.write(body.replace('"prefix_rows":', '"prefix_rows": NaN, "x_":', 1))
            f["sha256"] = nullcheck._replay.sha256_file(path)          # hash-matching but malformed
            self._rewrite(out, rec)
            os.environ["NC_MARK"] = marker
            try:
                errs = nullcheck.validate(out, t)
                res = nullcheck.replay(out, root=t)
            finally:
                del os.environ["NC_MARK"]
            self.assertTrue(any("reproducer.json rejected" in e for e in errs), errs)
            self.assertEqual(res["outcome"], "REFUSED")
            self.assertFalse(os.path.exists(marker))

    def test_R4_emitted_record_validated_at_runtime(self):
        from unittest import mock
        with tempfile.TemporaryDirectory() as t:
            out = os.path.join(t, "rec")
            with mock.patch.object(nullcheck._replay, "validate_record", return_value=["semantic: injected"]):
                with self.assertRaises(nullcheck._replay.EmissionError):
                    nullcheck.check(in_root(t, "features", "leak_next_bar"), make_ohlcv(120, 1), out=out, root=t)
            self.assertNotIn("replay.json", os.listdir(out))
            self.assertIn("EMISSION_INVALID.json", os.listdir(out))


class TestRecord(unittest.TestCase):
    df = make_ohlcv(300, 4)

    def test_record_valid_private_and_replays(self):
        with tempfile.TemporaryDirectory() as t:
            out = os.path.join(t, "rec")
            nullcheck.check(in_root(t, "features", "leak_whole_sample_z"), self.df, out=out, root=t)
            self.assertEqual(stat.S_IMODE(os.stat(out).st_mode), 0o700)
            for dp, dn, fn in os.walk(out):
                for f in fn:
                    self.assertEqual(stat.S_IMODE(os.stat(os.path.join(dp, f)).st_mode), 0o600)
            self.assertEqual(nullcheck.validate(out, t), [])
            rec = _load_json(os.path.join(out, "replay.json"))
            self.assertEqual(rec["schema_version"], "0.4.0")
            self.assertEqual(rec["status"], "DIVERGENCE_OBSERVED")
            self.assertEqual(rec["reproduction"]["availability"], "AVAILABLE", rec["reproduction"]["reason"])
            self.assertTrue(all(not os.path.isabs(f["path"]) for f in rec["files"] if f["path"]))
            self.assertIn("code/features.py", [f["path"] for f in rec["files"]])
            res = nullcheck.replay(out, root=t)
            self.assertEqual(res["outcome"], "REPRODUCED", res)
            self.assertEqual(res["small_reproducer"], "REPRODUCED")

    def test_source_outside_root_is_unavailable(self):
        with tempfile.TemporaryDirectory() as t:
            out = os.path.join(t, "rec")
            nullcheck.check(LEAKS["leak_next_bar"], self.df, out=out, root=t)
            rec = _load_json(os.path.join(out, "replay.json"))
            self.assertEqual(rec["reproduction"]["availability"], "UNAVAILABLE")
            self.assertIn("outside the chosen root", rec["reproduction"]["reason"])
            self.assertEqual(nullcheck.validate(out, t), [])
            self.assertEqual(nullcheck.replay(out, root=t)["outcome"], "UNAVAILABLE")

    def test_clean_and_not_assessed_records_validate(self):
        with tempfile.TemporaryDirectory() as t:
            f = in_root(t, "features", "clean_sma_cross")
            nullcheck.check(f, self.df, out=os.path.join(t, "a"), root=t)
            nullcheck.check(in_root(t, "features", "clean_long_warmup"), self.df, out=os.path.join(t, "b"), root=t)
            nullcheck.check(in_root(t, "features", "unseeded_noise"), self.df, out=os.path.join(t, "c"), root=t)
            for k, want in (("a", "NO_DIVERGENCE_OBSERVED"), ("b", "NO_DIVERGENCE_OBSERVED"), ("c", "NOT_ASSESSED")):
                self.assertEqual(nullcheck.validate(os.path.join(t, k), t), [], k)
                self.assertEqual(_load_json(os.path.join(t, k, "replay.json"))["status"], want)
            self.assertNotIn("reproducer.csv", os.listdir(os.path.join(t, "a")))
            self.assertEqual(nullcheck.replay(os.path.join(t, "a"), root=t)["outcome"], "REPRODUCED")

    def test_tampered_record_refused(self):
        with tempfile.TemporaryDirectory() as t:
            out = os.path.join(t, "rec")
            nullcheck.check(in_root(t, "features", "leak_next_bar"), self.df, out=out, root=t)
            with open(os.path.join(t, "code", "features.py"), "a") as f:
                f.write("\n# edited\n")
            self.assertEqual(nullcheck.replay(out, root=t)["outcome"], "REFUSED")
            self.assertTrue(nullcheck.replay(out, root=t, allow_modified=True)["outcome"].startswith("MODIFIED_"))

    def test_validator_catches_bad_records(self):
        with tempfile.TemporaryDirectory() as t:
            out = os.path.join(t, "rec")
            nullcheck.check(in_root(t, "features", "leak_next_bar"), self.df, out=out, root=t)
            rec = _load_json(os.path.join(out, "replay.json"))
        V = nullcheck._replay.validate_record
        self.assertEqual(V(rec), [])
        bad = json.loads(json.dumps(rec)); bad["status"] = "NO_DIVERGENCE_OBSERVED"
        self.assertTrue(any("precedence" in e for e in V(bad)))
        bad = json.loads(json.dumps(rec)); bad["files"][0]["path"] = "/etc/passwd"
        self.assertTrue(V(bad))
        bad = json.loads(json.dumps(rec)); bad["probes"][0]["compared_ranges"][0]["stop"] += 1
        self.assertTrue(V(bad))
        bad = json.loads(json.dumps(rec)); fd = next(p for p in bad["probes"] if p["first_divergence"])["first_divergence"]
        fd["full_state"] = "NAN"
        self.assertTrue(V(bad))

    def test_strict_json_rejects_duplicates_and_nonfinite(self):
        with tempfile.TemporaryDirectory() as t:
            out = os.path.join(t, "rec")
            nullcheck.check(in_root(t, "features", "leak_next_bar"), self.df, out=out, root=t)
            path = os.path.join(out, "replay.json")
            with open(path) as f:
                good = f.read()
            os.chmod(path, 0o600)
            for bad, word in ((good.replace('"status":', '"status": "ERROR", "status":', 1), "duplicate"),
                              (good.replace('"atol": 1e-12', '"atol": NaN', 1), "NaN"),
                              (good.replace('"rtol": 1e-09', '"rtol": Infinity', 1), "Infinity")):
                with open(path, "w") as f:
                    f.write(bad)
                with self.assertRaises(ValueError) as cm:
                    nullcheck.validate(out, t)
                self.assertIn(word, str(cm.exception))
                with self.assertRaises(ValueError):
                    nullcheck.replay(out, root=t)
            deep = os.path.join(t, "deep.json")
            with open(deep, "w") as f:
                f.write("[" * 17 + "]" * 17)
            with self.assertRaises(ValueError):
                nullcheck._replay.strict_load(deep)

    def test_lambda_target_is_unavailable_not_invented(self):
        with tempfile.TemporaryDirectory() as t:
            out = os.path.join(t, "rec")
            nullcheck.check(lambda d: d["Close"].shift(-1), self.df, out=out, root=t)  # noqa: E731
            rec = _load_json(os.path.join(out, "replay.json"))
            self.assertEqual(rec["reproduction"]["availability"], "UNAVAILABLE")
            self.assertIn("lambda", rec["reproduction"]["reason"])
            self.assertEqual(nullcheck.validate(out, t), [])

    def test_refuses_existing_folder(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(FileExistsError):
                nullcheck.check(LEAKS["leak_next_bar"], self.df, out=t)


class TestCLI(unittest.TestCase):
    def run_cli(self, *args):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
            code = main(list(args))
        return code, buf.getvalue()

    def test_exit_codes(self):
        with tempfile.TemporaryDirectory() as t:
            data = os.path.join(t, "raw.csv")
            make_ohlcv(250, 5).to_csv(data)
            in_root(t, "features", "clean_rsi")
            feat = os.path.join(t, "code", "features.py")
            code, out = self.run_cli("check", f"{feat}:leak_centred_mean", "--data", data, "--out", os.path.join(t, "r"), "--root", t)
            self.assertEqual(code, 1, out)
            self.assertIn("DIVERGENCE", out)
            self.assertEqual(self.run_cli("replay", os.path.join(t, "r"), "--root", t)[0], 0)
            self.assertEqual(self.run_cli("validate", os.path.join(t, "r"), "--root", t)[0], 0)
            self.assertEqual(self.run_cli("check", f"{feat}:clean_rsi", "--data", data, "--warmup", "15")[0], 0)
            self.assertEqual(self.run_cli("check", f"{feat}:clean_rsi", "--data", data)[0], 0)
            self.assertEqual(self.run_cli("check", f"{feat}:unseeded_noise", "--data", data)[0], 3)
            self.assertEqual(self.run_cli("check", f"{feat}:no_such", "--data", data)[0], 2)
            self.assertEqual(self.run_cli("check", "nocolon", "--data", data)[0], 2)
            rec = _load_json(os.path.join(t, "r", "replay.json"))
            orig = next(f for f in rec["files"] if f["id"] == "original_input")
            self.assertEqual(orig["path"], "raw.csv")

    def test_replay_in_new_interpreter_from_other_cwd(self):
        with tempfile.TemporaryDirectory() as t:
            out = os.path.join(t, "rec")
            nullcheck.check(in_root(t, "features", "leak_weekly_bfill"), make_ohlcv(200, 8), out=out, root=t)
            env = dict(os.environ, PYTHONPATH=ROOT)
            p = subprocess.run([sys.executable, "-m", "nullcheck", "replay", "rec"], cwd=t, env=env,
                               capture_output=True, text=True, timeout=120)
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
            self.assertEqual(json.loads(p.stdout)["outcome"], "REPRODUCED")


@unittest.skipUnless(HAVE_BT, "backtesting not installed")
class TestBacktestingAdapter(unittest.TestCase):
    df = make_ohlcv(400, 6)

    def test_clean_strategy_with_end_liquidation(self):
        for kw in ({}, {"finalize_trades": True}):
            r = nullcheck.check_backtesting(B.CleanSmaCross, self.df, backtest_kwargs=kw, settings=S(warmup_rows=40))
            self.assertEqual(r.status, nullcheck.NO_DIVERGENCE, r.divergences[:2])
            self.assertEqual(len(r.channels), 2)

    def test_leaks_inside_strategy(self):
        for St, ch in ((B.LeakShiftIndicator, "I[2]:next_close"), (B.LeakAttrZscore, "attr:z"), (B.LeakDfRank, "I[2]:rank")):
            with self.subTest(St.__name__):
                r = nullcheck.check_backtesting(St, self.df, settings=S(warmup_rows=40))
                self.assertEqual(r.status, nullcheck.DIVERGENCE)
                self.assertIn(ch, {d["channel"] for d in r.divergences})

    def test_prepare_recomputed_and_index_checked(self):
        r = nullcheck.check_backtesting(B.UsesPreparedZ, self.df, prepare=B.prepare_leaky_z, settings=S(warmup_rows=40))
        self.assertEqual(r.status, nullcheck.DIVERGENCE)
        self.assertIn("data:z", r.channels)
        r = core.check(lambda d: {"z": B.prepare_leaky_z(d)["z"]}, self.df)   # the prepared column alone also diverges
        self.assertEqual(r.status, nullcheck.DIVERGENCE)
        r = nullcheck.check_backtesting(B.UsesPreparedZ, self.df, prepare=B.prepare_clean_z, settings=S(warmup_rows=40))
        self.assertEqual(r.status, nullcheck.NO_DIVERGENCE)
        r = nullcheck.check_backtesting(B.UsesPreparedZ, self.df, prepare=lambda d: B.prepare_clean_z(d).iloc[::-1])
        self.assertEqual(r.status, nullcheck.NOT_ASSESSED)

    def test_precomputed_column_never_clean(self):
        r = nullcheck.check_backtesting(B.UsesPreparedZ, B.prepare_clean_z(self.df), settings=S(warmup_rows=40))
        self.assertEqual(r.status, nullcheck.NOT_ASSESSED)
        self.assertFalse(r.raw_prefix_recomputed)

    def test_backtesting_record_replays(self):
        with tempfile.TemporaryDirectory() as t:
            out = os.path.join(t, "rec")
            St, prep = in_root(t, "bt_strategies", "UsesPreparedZ"), in_root(t, "bt_strategies", "prepare_leaky_z")
            nullcheck.check_backtesting(St, self.df, prepare=prep, out=out, root=t)
            self.assertEqual(nullcheck.validate(out, t), [])
            rec = _load_json(os.path.join(out, "replay.json"))
            self.assertEqual(rec["adapter"]["framework_version"], "0.6.6")
            self.assertEqual(len([f for f in rec["files"] if f["role"] == "SOURCE"]), 1)
            self.assertEqual(nullcheck.replay(out, root=t)["outcome"], "REPRODUCED")


if __name__ == "__main__":
    unittest.main()

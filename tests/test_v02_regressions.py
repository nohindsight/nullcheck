"""Regression oracles from blind-suite misses M1--M6 and the real-world TrailingStrategy back-fill report.
No new held-out evaluation cases were used to write these tests.
"""
import json
import os
import tempfile
import unittest
import numpy as np
import pandas as pd
import nullcheck
from nullcheck import core
from fixtures import v02_features as F
from tests.test_nullcheck import in_root, HAVE_BT

S = nullcheck.Settings


class DisclosedRepros(unittest.TestCase):
    def test_M1_bfill_is_audited_even_with_warmup(self):
        for warmup in (0, 20, 1000):
            r = nullcheck.check(F.warmup_bfill, F.prices(), settings=S(warmup_rows=warmup))
            self.assertEqual(r.status, nullcheck.DIVERGENCE)
            self.assertEqual(r.divergences[0]["kind"], "MISSINGNESS")
            self.assertEqual(r.divergences[0]["row"], 0)
            self.assertIn(1, [p["prefix_length"] for p in r.probes])

    def test_M2_zero_bar_and_halt_cutoffs_across_offsets(self):
        raw = F.prices()
        raw.iloc[337, :4] = 0.
        for offset in (0, 1, 7, 19):
            d = raw.iloc[offset:]
            cuts = core.choose_cutoffs(len(d), S(), d)
            self.assertIn(338-offset, cuts)
            self.assertEqual(nullcheck.check(F.zero_bfill, d).status, nullcheck.DIVERGENCE)
        raw = F.prices()
        raw.iloc[337, raw.columns.get_loc("Volume")] = 0.
        self.assertEqual(nullcheck.check(F.halt_bfill, raw).status, nullcheck.DIVERGENCE)

    def test_M3_month_end_future_price(self):
        for offset in (0, 1, 7, 19):
            d = F.prices().iloc[offset:]
            self.assertEqual(nullcheck.check(F.month_end_future, d).status, nullcheck.DIVERGENCE)

    def test_M4_descending_refused_before_callback(self):
        for f in (F.descending_lag, F.descending_causal):
            r = nullcheck.check(f, F.prices(300).iloc[::-1], settings=S(warmup_rows=1))
            self.assertEqual(r.status, nullcheck.NOT_ASSESSED)
            self.assertEqual(r.runs_used, 0)
            self.assertIn("chronological", r.reason)
        raw = F.prices(40)
        for d in (raw.sample(frac=1, random_state=1), pd.concat([raw, raw.iloc[[-1]]])):
            self.assertEqual(nullcheck.check(F.descending_lag, d).runs_used, 0)

    def test_M5_invalid_or_unstable_channels_do_not_mask_z(self):
        for f, excluded in ((F.mixed_text, "sym"), (F.mixed_random, "rnd"), (F.leak_with_inf, "bad")):
            r = nullcheck.check(f, F.prices())
            self.assertEqual(r.status, nullcheck.DIVERGENCE)
            self.assertEqual({d["channel"] for d in r.divergences}, {"z"})
            self.assertTrue(any(x["channel"] == excluded for p in r.probes for x in p["skipped_ranges"]))
            self.assertTrue(all(p["reason"] for p in r.probes if p["state"] == "COMPARED"))

    def test_nan_warmup_sparse_and_all_nan_channel_do_not_block_numeric_coverage(self):
        for f in (F.sparse_clean, F.long_warmup):
            r = nullcheck.check(f, F.prices())
            self.assertEqual(r.status, nullcheck.NO_DIVERGENCE)
            self.assertTrue(any(p["state"] == "SKIPPED_BOTH_NAN" for p in r.probes))
        r = nullcheck.check(F.long_warmup, F.prices(100))
        self.assertEqual(r.status, nullcheck.NOT_ASSESSED)  # zero coverage is never success

    def test_invalid_cells_do_not_erase_same_channel_finding(self):
        r = core.compare_channel("x", np.array([np.inf, 9.]), np.array([np.inf, 3.]), 2, S())
        self.assertEqual(r["diffs"][0]["row"], 1)
        self.assertIsNotNone(r["unassessable"])

    def test_stale_and_changepoint_boundaries_and_cap(self):
        d = F.prices()
        d.iloc[337, :4] = d.iloc[336, :4]
        self.assertIn(338, core.choose_cutoffs(len(d), S(), d))
        d.iloc[400:, :4] *= 3
        self.assertIn(400, core.choose_cutoffs(len(d), S(), d))
        many = F.prices(10000)
        many.iloc[::2, :4] = 0
        cuts = core.choose_cutoffs(len(many), S(n_cutoffs=1000), many)
        self.assertLessEqual(len(cuts), 128)
        self.assertEqual(cuts, sorted(set(cuts)))
        self.assertIn(1, cuts)
        self.assertIn(len(many)-1, cuts)

    def test_partial_and_missing_records_replay(self):
        for name in ("warmup_bfill", "mixed_text", "sparse_clean", "long_warmup", "leak_with_inf", "mixed_random"):
            with self.subTest(name), tempfile.TemporaryDirectory() as t:
                fn = in_root(t, "v02_features", name)
                out = os.path.join(t, "record")
                nullcheck.check(fn, F.prices(240), settings=S(warmup_rows=20), out=out, root=t)
                self.assertEqual(nullcheck.validate(out, t), [])
                self.assertEqual(nullcheck.replay(out, root=t)["outcome"], "REPRODUCED")

    def test_refused_order_record_replays(self):
        with tempfile.TemporaryDirectory() as t:
            fn = in_root(t, "v02_features", "descending_lag")
            out = os.path.join(t, "record")
            nullcheck.check(fn, F.prices(200).iloc[::-1], out=out, root=t)
            self.assertEqual(nullcheck.validate(out, t), [])
            self.assertEqual(nullcheck.replay(out, root=t)["outcome"], "REPRODUCED")

    def test_new_skip_state_cannot_hide_arbitrary_omission(self):
        with tempfile.TemporaryDirectory() as t:
            out = os.path.join(t, "record")
            nullcheck.check(in_root(t, "v02_features", "long_warmup"), F.prices(240), out=out, root=t)
            with open(os.path.join(out, "replay.json")) as f:
                rec = json.load(f)
        p = next(p for p in rec["probes"] if p["state"] == "SKIPPED_BOTH_NAN")
        p["skipped_ranges"][0]["reason"] = "arbitrary omitted data"
        self.assertTrue(nullcheck._replay.validate_record(rec))

    def test_new_unsupported_prefix_channel_is_disclosed(self):
        raw = F.prices(100)
        def features(d):
            out = {"price": d.Close}
            if len(d) < len(raw):
                out["extra_text"] = np.repeat("unsupported", len(d))
            return out
        r = nullcheck.check(features, raw)
        self.assertEqual(r.status, nullcheck.NOT_ASSESSED)
        self.assertTrue(all("extra_text" in p["reason"] for p in r.probes))

    def test_old_format_range_validator_is_retained(self):
        p = {"prefix_length": 10, "state": "COMPARED", "compared_ranges": [{"channel": "x", "start": 3, "stop": 10}],
             "skipped_ranges": [{"channel": "x", "start": 0, "stop": 3, "reason": core.R_DECLARED}]}
        ct = {"warmup_rows": 3, "terminal_excluded_rows": 0, "warmup_policy": "DECLARED_AND_LEADING_BOTH_NAN"}
        self.assertEqual(nullcheck._replay._probe_ranges_ok(p, ["x"], ct), [])
        p["compared_ranges"][0]["start"] = 2
        self.assertTrue(nullcheck._replay._probe_ranges_ok(p, ["x"], ct))


@unittest.skipUnless(HAVE_BT, "backtesting not installed")
class AdapterDisclosedRepros(unittest.TestCase):
    def test_M6_scalar_and_list_each_detected(self):
        from fixtures import v02_bt as B
        for strategy, prefix in ((B.ScalarLeak, "attr:thr"), (B.ListLeak, "attr:levels["), (B.M6, "attr:")):
            r = nullcheck.check_backtesting(strategy, F.prices(), settings=S(warmup_rows=10))
            self.assertEqual(r.status, nullcheck.DIVERGENCE)
            self.assertTrue(any(d["channel"].startswith(prefix) for d in r.divergences))

    def test_constants_and_private_raw_copy_are_insufficient(self):
        from fixtures import v02_bt as B
        r = nullcheck.check_backtesting(B.ConstantState, F.prices())
        self.assertEqual(r.status, nullcheck.NOT_ASSESSED)
        self.assertEqual(r.reason, "insufficient meaningful coverage")
        self.assertIn("attr:_private", r.channels)
        self.assertIn("attr:levels[2]", r.channels)

    def test_realworld_private_ATR_sparse_signal_and_declared_warmup(self):
        from fixtures import v02_bt as B
        for warmup in (0, 100):
            r = nullcheck.check_backtesting(B.TrailingSparse, F.prices(), settings=S(warmup_rows=warmup))
            self.assertEqual(r.status, nullcheck.DIVERGENCE)
            self.assertIn("attr:_TrailingStrategy__atr", r.channels)
            self.assertTrue(any(d["channel"] == "attr:_TrailingStrategy__atr" and d["row"] == 0 for d in r.divergences))

    def test_realworld_private_trailing_pct_scalar_is_captured(self):
        from fixtures import v02_bt as B
        from nullcheck.adapters.backtesting_py import make_feature_fn
        fn = make_feature_fn(B.TrailingPct, None, {}, {})
        a, b = fn(F.prices()), fn(F.prices().iloc[:200])
        key = "attr:_TrailingStrategy__n_atr"
        self.assertIn(key, a)
        self.assertTrue(core.compare_channel(key, a[key], b[key], 200, S())["diffs"])

    def test_row_local_sparse_signal_and_next_only_mutation_are_insufficient(self):
        from fixtures import v02_bt as B
        for st in (B.SparseSignal, B.NextOnlyMutation):
            raw = F.prices()
            raw.iloc[::10, raw.columns.get_loc("Open")] *= .9
            r = nullcheck.check_backtesting(st, raw)
            self.assertEqual(r.status, nullcheck.NOT_ASSESSED)
            self.assertEqual(r.reason, "insufficient meaningful coverage")

    def test_private_ATR_record_replays(self):
        with tempfile.TemporaryDirectory() as t:
            out = os.path.join(t, "record")
            st = in_root(t, "v02_bt", "TrailingSparse")
            nullcheck.check_backtesting(st, F.prices(), settings=S(warmup_rows=100), out=out, root=t)
            self.assertEqual(nullcheck.validate(out, t), [])
            self.assertEqual(nullcheck.replay(out, root=t)["outcome"], "REPRODUCED")

import json
import os
import tempfile
import unittest
import numpy as np
import nullcheck
from nullcheck import core, _replay as replay
from fixtures import v02_features as D, v023_features as F
from tests.test_nullcheck import HAVE_BT, in_root

FAST = nullcheck.Settings(n_cutoffs=6, reduce_budget=0)

class MinimumCoverage(unittest.TestCase):
    def assert_insufficient(self, result):
        self.assertEqual(result.status, nullcheck.NOT_ASSESSED)
        self.assertEqual(result.reason, "insufficient meaningful coverage")
        self.assertTrue(result.channels)
        self.assertTrue(any(p["compared_ranges"] for p in result.probes))
        self.assertTrue(all(p["reason"] == result.reason for p in result.probes if p["state"] == "COMPARED"))
        rendered = nullcheck.render(result)
        self.assertIn("STATUS: NOT_ASSESSED", rendered)
        self.assertTrue(all("channel  " + c in rendered for c in result.channels))
        self.assertLessEqual(result.runs_used, result.max_runs)

    def test_raw_constant_and_affine_counter_callbacks_are_insufficient(self):
        for fn in (F.passthrough, F.constant, F.counters):
            with self.subTest(fn=fn.__name__):
                result = nullcheck.check(fn, D.prices(40), settings=FAST)
                self.assert_insufficient(result)
                self.assertTrue(result.raw_prefix_recomputed)
                self.assertEqual(result.runs_used, 2 + 2*len(result.probes))

    def test_generated_nonlinear_sequence_does_not_establish_input_dependency(self):
        result = nullcheck.check(F.nonlinear_counter, D.prices(40), settings=FAST)
        self.assert_insufficient(result)
        self.assertTrue(any("no observed dependency" in s for s in result.limitations))
        self.assertEqual(result.runs_used, 4 + 2*len(result.probes))

    def test_derived_causal_channel_qualifies_among_trivial_channels(self):
        raw = D.prices(40)
        original = raw.copy(deep=True)
        result = nullcheck.check(F.causal_mixed, raw, settings=FAST)
        self.assertEqual(result.status, nullcheck.NO_DIVERGENCE, result.reason)
        self.assertEqual(result.channels, ["constant", "counter", "raw", "signal"])
        self.assertTrue(raw.equals(original))
        self.assertEqual(result.runs_used, 4 + 2*len(result.probes))
        self.assertTrue(any("'signal'" in s and "non-trivial" in s for s in result.limitations))

    def test_sparse_data_dependent_events_can_qualify(self):
        result = nullcheck.check(F.sparse_events, D.prices(80), settings=FAST)
        self.assertEqual(result.status, nullcheck.NO_DIVERGENCE, result.reason)
        self.assertTrue(any(p["skipped_ranges"] for p in result.probes))

    def test_generated_sparse_events_do_not_establish_dependency(self):
        result = nullcheck.check(F.generated_sparse, D.prices(80), settings=FAST)
        self.assert_insufficient(result)

    def test_older_record_validates_but_requires_original_runner(self):
        with tempfile.TemporaryDirectory() as t:
            fn = in_root(t, "v023_features", "constant")
            out = os.path.join(t, "record")
            nullcheck.check(fn, D.prices(40), settings=FAST, out=out, root=t)
            path = os.path.join(out, "replay.json")
            record = replay.strict_load(path)
            # Emulate the valid v0.2.2 semantics without changing any frozen file.
            record["check"]["version"] = record["check"]["runner_version"] = "0.2.2"
            record["adapter"]["version"] = "0.2.2"
            record["coverage"]["max_callback_runs"] -= 2
            record["status"], record["reason"] = nullcheck.NO_DIVERGENCE, None
            for probe in record["probes"]: probe["reason"] = None
            with open(path, "w") as f: json.dump(record, f)
            self.assertEqual(nullcheck.validate(out, t), [])
            self.assertEqual(nullcheck.replay(out, root=t)["outcome"], "UNSUPPORTED_RUNNER")

    def test_unrelated_all_missing_column_does_not_block_dependency(self):
        raw = D.prices(40)
        raw["Volume"] = np.nan
        result = nullcheck.check(F.causal, raw, settings=FAST)
        self.assertEqual(result.status, nullcheck.NO_DIVERGENCE, result.reason)
        self.assertTrue(raw.Volume.isna().all())

    def test_uncompared_variation_cannot_satisfy_minimum(self):
        raw = D.prices(10)
        full = {"tail_only": np.r_[np.ones(8), 2., 5.]}
        probes = [{"skipped_ranges": [], "compared_ranges": [{"channel": "tail_only", "start": 0, "stop": 8}]}]
        eligible, reasons = core.meaningful_coverage(full, raw, probes, FAST)
        self.assertEqual(eligible, [])
        self.assertIn("constant", reasons["tail_only"])

    def test_divergence_precedes_coverage_gate(self):
        result = nullcheck.check(F.constant_and_leak, D.prices(40), settings=FAST)
        self.assertEqual(result.status, nullcheck.DIVERGENCE)
        self.assertTrue(result.divergences)
        self.assertEqual(result.runs_used, 2 + 2*len(result.probes))

    def test_dependency_probe_failure_is_inconclusive_not_a_leak_or_crash(self):
        raw = D.prices(40)
        for key in ("Open", "High", "Low", "Close"):
            raw[key] = np.arange(1., 41.)**2
        result = nullcheck.check(F.dependency_failure, raw, settings=FAST)
        self.assert_insufficient(result)
        self.assertTrue(any("dependency probe unsupported" in s for s in result.limitations))

    def test_insufficient_and_qualified_records_validate_and_replay(self):
        for name in ("passthrough", "constant", "counters", "nonlinear_counter", "causal_mixed"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as t:
                fn = in_root(t, "v023_features", name)
                out = os.path.join(t, "record")
                nullcheck.check(fn, D.prices(40), settings=FAST, out=out, root=t)
                self.assertEqual(nullcheck.validate(out, t), [])
                self.assertEqual(nullcheck.replay(out, root=t)["outcome"], "REPRODUCED")

    def test_forged_success_is_rejected_by_existing_schema_semantics(self):
        with tempfile.TemporaryDirectory() as t:
            fn = in_root(t, "v023_features", "constant")
            out = os.path.join(t, "record")
            nullcheck.check(fn, D.prices(40), settings=FAST, out=out, root=t)
            record = replay.strict_load(os.path.join(out, "replay.json"))
            record["status"], record["reason"] = nullcheck.NO_DIVERGENCE, None
            self.assertTrue(any("precedence" in e for e in replay.validate_record(record)))

    @unittest.skipUnless(HAVE_BT, "backtesting not installed")
    def test_T01_raw_precomputed_predictions_are_insufficient(self):
        from fixtures.v023_bt import RawPredictions
        raw = D.prices(40)
        raw["Preds"] = np.sin(np.arange(len(raw)))
        result = nullcheck.check_backtesting(RawPredictions, raw, settings=FAST)
        self.assert_insufficient(result)
        self.assertEqual(result.channels, ["attr:entry_bar", "attr:predictions"])
        self.assertFalse(result.raw_prefix_recomputed)
        self.assertTrue(any("pass-through" in s for s in result.limitations))

    @unittest.skipUnless(HAVE_BT, "backtesting not installed")
    def test_B04_counters_only_are_insufficient(self):
        from fixtures.v023_bt import CountersOnly
        result = nullcheck.check_backtesting(CountersOnly, D.prices(40), settings=FAST)
        self.assert_insufficient(result)
        self.assertEqual(len(result.channels), 4)

    @unittest.skipUnless(HAVE_BT, "backtesting not installed")
    def test_S10_one_constant_is_insufficient(self):
        from fixtures.v023_bt import ConstantOnly
        result = nullcheck.check_backtesting(ConstantOnly, D.prices(40), settings=FAST)
        self.assert_insufficient(result)
        self.assertEqual(result.channels, ["attr:dollar_amount"])

    @unittest.skipUnless(HAVE_BT, "backtesting not installed")
    def test_prepare_raw_copy_is_not_a_derived_feature(self):
        from fixtures.v023_bt import PreparedCopy
        result = nullcheck.check_backtesting(PreparedCopy, D.prices(40), prepare=F.prepare_copy, settings=FAST)
        self.assert_insufficient(result)
        self.assertEqual(result.channels, ["attr:copy", "data:prediction"])

    @unittest.skipUnless(HAVE_BT, "backtesting not installed")
    def test_strategy_coverage_records_replay(self):
        for name in ("RawPredictions", "CountersOnly", "ConstantOnly"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as t:
                strategy = in_root(t, "v023_bt", name)
                raw = D.prices(40)
                if name == "RawPredictions": raw["Preds"] = np.sin(np.arange(len(raw)))
                out = os.path.join(t, "record")
                self.assert_insufficient(nullcheck.check_backtesting(strategy, raw, settings=FAST, out=out, root=t))
                self.assertEqual(nullcheck.validate(out, t), [])
                self.assertEqual(nullcheck.replay(out, root=t)["outcome"], "REPRODUCED")

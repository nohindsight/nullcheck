import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import nullcheck
from nullcheck import core
from fixtures import v02_features as D
from fixtures import v022_features as F
from fixtures.v023_features import causal
from tests.test_nullcheck import HAVE_BT, in_root

FAST = nullcheck.Settings(n_cutoffs=4, reduce_budget=0)

class ExternalCoverage(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)/"predictions.csv"
        pd.DataFrame({"prediction": np.arange(1000.)}).to_csv(self.path, index=False)
        F.FILE = self.path
        self.addCleanup(setattr, F, "FILE", None)
        self.raw = D.prices(40)

    def assert_external(self, r, kind):
        self.assertEqual(r.status, nullcheck.NOT_ASSESSED, r.reason)
        channels = [c for c in r.channels if c.startswith("input:"+kind+":")]
        self.assertTrue(channels, r.channels)
        self.assertFalse(r.raw_prefix_recomputed)
        for channel in channels:
            for probe in r.probes:
                self.assertTrue(any(s["channel"] == channel for s in probe["skipped_ranges"]))
        label = "Default-argument state not assessed" if kind == "default" else "External precomputed inputs not assessed"
        self.assertIn(label, nullcheck.render(r))

    def test_runtime_file_is_per_input_unassessed(self):
        self.assert_external(nullcheck.check(F.external_file, self.raw, settings=FAST), "file")

    def test_data_file_named_py_is_still_unassessed(self):
        renamed = self.path.with_suffix(".py")
        renamed.write_bytes(self.path.read_bytes())
        F.FILE = renamed
        self.assert_external(nullcheck.check(F.external_file, self.raw, settings=FAST), "file")

    def test_lazy_import_is_not_an_external_data_input(self):
        def lazy(rows):
            import fractions
            return rows.Close * float(fractions.Fraction(2, 1))
        r = nullcheck.check(lazy, self.raw, settings=FAST)
        self.assertEqual(r.status, nullcheck.NOT_ASSESSED)  # row-local output
        self.assertEqual(r.reason, "insufficient meaningful coverage")
        self.assertFalse(any(c.startswith("input:") for c in r.channels))

    def test_module_array_closure_and_default_are_unassessed(self):
        precomputed = np.arange(1000.)
        def closure(rows): return precomputed[:len(rows)]
        def default(rows, fixed=precomputed): return fixed[:len(rows)]
        for fn in (F.external_global, closure, default):
            with self.subTest(fn=fn.__name__):
                self.assert_external(nullcheck.check(fn, self.raw, settings=FAST), "default" if fn is default else "memory")

    def test_read_only_seen_in_prefix_is_covered_at_every_probe(self):
        self.assert_external(nullcheck.check(F.prefix_file_only, self.raw, settings=FAST), "file")

    def test_external_input_does_not_erase_other_divergence(self):
        r = nullcheck.check(F.external_file_and_leak, self.raw, settings=FAST)
        self.assertEqual(r.status, nullcheck.DIVERGENCE)
        self.assertTrue(any(c.startswith("input:file:") for c in r.channels))
        self.assertTrue(r.divergences)

    def test_all_nan_with_external_has_valid_explicit_coverage(self):
        out = str(Path(self.tmp.name)/"record_nan")
        r = nullcheck.check(F.all_nan_file, self.raw, settings=FAST, out=out, root=self.tmp.name)
        self.assert_external(r, "file")
        self.assertEqual(nullcheck.validate(out, self.tmp.name), [])

    def test_external_record_is_valid_but_replay_unavailable(self):
        out = str(Path(self.tmp.name)/"record")
        nullcheck.check(F.external_file, self.raw, settings=FAST, out=out, root=self.tmp.name)
        self.assertEqual(nullcheck.validate(out, self.tmp.name), [])
        record = json.loads((Path(out)/"replay.json").read_text())
        self.assertIn("external precomputed inputs", record["reproduction"]["reason"])
        self.assertEqual(nullcheck.replay(out, root=self.tmp.name)["outcome"], "UNAVAILABLE")

    def test_clean_callback_is_not_contaminated_by_previous_file_read(self):
        nullcheck.check(F.external_file, self.raw, settings=FAST)
        self.assertEqual(nullcheck.check(causal, self.raw, settings=FAST).status, nullcheck.NO_DIVERGENCE)

    @unittest.skipUnless(HAVE_BT, "backtesting not installed")
    def test_prediction_array_in_strategy_params_is_unassessed(self):
        from fixtures.v022_bt import ParameterPredictions
        r = nullcheck.check_backtesting(ParameterPredictions, self.raw, params={"predictions": np.arange(1000.)}, settings=FAST)
        self.assert_external(r, "memory")
        self.assertIn("input:memory:parameter.predictions", r.channels)

    @unittest.skipUnless(HAVE_BT, "backtesting not installed")
    def test_strategy_and_prepare_file_reads_are_unassessed(self):
        from fixtures.v022_bt import ExternalStrategy, PreparedStrategy
        self.assert_external(nullcheck.check_backtesting(ExternalStrategy, self.raw, settings=FAST), "file")
        self.assert_external(nullcheck.check_backtesting(PreparedStrategy, self.raw, prepare=F.prepare_file, settings=FAST), "file")

class CompactChronology(unittest.TestCase):
    def test_compact_date_strings_and_integers_refused_before_callback(self):
        raw = D.prices(40).reset_index(drop=True)
        dates = pd.date_range("2021-01-01", periods=len(raw)).strftime("%Y%m%d")
        for name in ("trade_date", "custom_key"):
            for values in (dates, dates.astype(int)):
                for reverse in (False, True):
                    with self.subTest(name=name, dtype=values.dtype, reverse=reverse):
                        raw[name] = values[::-1] if reverse else values
                        r = nullcheck.check(F.identity, raw, settings=FAST)
                        self.assertEqual(r.status, nullcheck.NOT_ASSESSED)
                        if reverse: self.assertEqual(r.runs_used, 0)
                        else: self.assertEqual(r.reason, "insufficient meaningful coverage")
                raw = raw.drop(columns=[name])

    def test_epoch_time_keys_all_units_and_strings(self):
        raw = D.prices(40).reset_index(drop=True)
        for scale in (1, 1000, 1000000, 1000000000):
            epochs = (1609459200 + np.arange(len(raw))*86400) * scale
            for values in (epochs, epochs.astype(str)):
                for reverse in (False, True):
                    raw["event_timestamp"] = values[::-1] if reverse else values
                    r = nullcheck.check(F.identity, raw, settings=FAST)
                    self.assertEqual(r.status, nullcheck.NOT_ASSESSED)
                    if reverse: self.assertEqual(r.runs_used, 0)
                    else: self.assertEqual(r.reason, "insufficient meaningful coverage")

    def test_epoch_aliases_are_temporal_keys(self):
        raw = D.prices(4).reset_index(drop=True)
        for key in ("ts", "unix_ms", "epoch_seconds"):
            raw[key] = [1609459203000, 1609459202000, 1609459201000, 1609459200000]
            r = nullcheck.check(F.identity, raw, settings=FAST)
            self.assertEqual(r.status, nullcheck.NOT_ASSESSED)
            self.assertEqual(r.runs_used, 0)
            raw = raw.drop(columns=[key])

    def test_invalid_duplicate_and_missing_compact_dates_refused(self):
        raw = D.prices(4).reset_index(drop=True)
        for values in ([20200101,20200102,20200102,20200104], [20200101,20200102,20200103,None], [20200101,20200102,20200103,20200230]):
            raw["trade_date"] = values
            r = nullcheck.check(F.identity, raw, settings=FAST)
            self.assertEqual(r.status, nullcheck.NOT_ASSESSED)
            self.assertEqual(r.runs_used, 0)

@unittest.skipUnless(HAVE_BT, "backtesting not installed")
class StateRegressions(unittest.TestCase):
    def test_tuple_row_arrays_are_capped_before_broadcast(self):
        from fixtures.v022_bt import TupleRows
        from nullcheck.adapters.backtesting_py import _snapshot
        from types import SimpleNamespace
        from backtesting._util import _Indicator
        raw = D.prices(10000)
        fake = SimpleNamespace(_indicators=[], bands=(raw.Close.to_numpy(), raw.Close.to_numpy()), causal=raw.Close.to_numpy())
        with patch("nullcheck.adapters.backtesting_py.np.full", side_effect=AssertionError("broadcast must not be attempted")):
            snap = _snapshot(fake, len(raw), _Indicator, {})
        self.assertEqual(set(snap), {"attr:causal"})
        self.assertIn("state expansion cap", snap.unassessed["attr:bands"])
        r = nullcheck.check_backtesting(TupleRows, raw, settings=FAST)
        self.assertEqual(r.status, nullcheck.NOT_ASSESSED)
        self.assertIn("state expansion cap", "\n".join(r.limitations))
        self.assertLessEqual(len(r.channels), 2)

    def test_string_none_categorical_comparisons_preserve_numeric_coverage(self):
        from fixtures.v022_bt import CategoricalState, CategoricalAndLeak
        raw = D.prices(40)
        r = nullcheck.check_backtesting(CategoricalState, raw, settings=FAST)
        self.assertEqual(r.status, nullcheck.NO_DIVERGENCE, r.reason)
        self.assertIn("attr:causal", r.channels)
        for name in ("mode", "last_signal", "labels", "names", "category_series", "object_strings", "object_array"):
            self.assertTrue(any(c == "attr:"+name or c.startswith("attr:"+name+"[") for c in r.categorical_channels), name)
        r = nullcheck.check_backtesting(CategoricalAndLeak, raw, settings=FAST)
        self.assertEqual(r.status, nullcheck.DIVERGENCE)

    def test_categorical_comparison_record_replays(self):
        with tempfile.TemporaryDirectory() as t:
            strategy = in_root(t,"v022_bt","CategoricalState")
            out = os.path.join(t,"record")
            nullcheck.check_backtesting(strategy, D.prices(40), settings=FAST, out=out, root=t)
            self.assertEqual(nullcheck.validate(out,t),[])
            self.assertEqual(nullcheck.replay(out,root=t)["outcome"],"REPRODUCED")


@unittest.skipUnless(HAVE_BT, "backtesting not installed")
class RealWorldBreakdownRegressions(unittest.TestCase):
    def test_numeric_300_by_4_state_emits_valid_unassessed_record(self):
        with tempfile.TemporaryDirectory() as t:
            strategy = in_root(t, "v022_bt", "NumericLookupTable")
            out = os.path.join(t, "record")
            r = nullcheck.check_backtesting(strategy, D.prices(400), settings=FAST, out=out, root=t)
            self.assertEqual(r.status, nullcheck.NOT_ASSESSED)
            self.assertIn("attr:table", r.channels)
            self.assertLess(len(r.channels), 1000)
            self.assertEqual(nullcheck.validate(out,t), [])
            self.assertEqual(nullcheck.replay(out,root=t)["outcome"], "REPRODUCED")

    def test_future_datetime_values_diverge_empty_containers_stay_excluded(self):
        from fixtures.v022_bt import DatetimeMetadata, EmptyToPopulated
        r = nullcheck.check_backtesting(DatetimeMetadata, D.prices(40), settings=FAST)
        self.assertEqual(r.status, nullcheck.DIVERGENCE, r.reason)
        for name in ("lastday", "timestamp"):
            self.assertIn("attr:"+name, r.categorical_channels)
        for name in ("events", "days"):
            self.assertTrue(any(repr(name) in line and "EXCLUDED" in line for line in r.limitations))
        r = nullcheck.check_backtesting(EmptyToPopulated, D.prices(40), settings=FAST)
        self.assertEqual(r.status, nullcheck.DIVERGENCE)  # datetime finding wins over partial container coverage
        self.assertTrue(any("empty metadata container changed" in line for line in r.limitations))

    def test_prepare_metadata_columns_compared_numeric_signal_preserved(self):
        from fixtures.v022_bt import ScalarOnly, prepare_metadata
        r = nullcheck.check_backtesting(ScalarOnly, D.prices(40), prepare=prepare_metadata, settings=FAST)
        self.assertEqual(r.status, nullcheck.NO_DIVERGENCE, r.reason)
        self.assertIn("data:signal", r.channels)
        for name in ("Local time", "date"):
            self.assertIn("data:"+name, r.categorical_channels)
            self.assertTrue(any(x["channel"] == "data:"+name for p in r.probes for x in p["compared_ranges"]))
        self.assertNotIn("init-state only", nullcheck.render(r))

    def test_scalar_only_agreement_is_insufficient(self):
        from fixtures.v022_bt import ScalarOnly
        r = nullcheck.check_backtesting(ScalarOnly, D.prices(40), settings=FAST)
        self.assertEqual(r.status, nullcheck.NOT_ASSESSED)
        self.assertEqual(r.reason, "insufficient meaningful coverage")
        self.assertIn("STATUS: NOT_ASSESSED", nullcheck.render(r))
        self.assertTrue(any(line.startswith("SNAPSHOT_SCOPE=INIT_STATE_ONLY:") for line in r.limitations))

    def test_mixed_frame_metadata_exclusion_never_excuses_wrong_index(self):
        from fixtures.v022_bt import MixedFrame, WrongIndexFrame
        r = nullcheck.check_backtesting(MixedFrame, D.prices(40), settings=FAST)
        self.assertEqual(r.status, nullcheck.NO_DIVERGENCE, r.reason)
        self.assertIn("attr:frame.Close", r.channels)
        r = nullcheck.check_backtesting(WrongIndexFrame, D.prices(40), settings=FAST)
        self.assertEqual(r.status, nullcheck.NOT_ASSESSED)
        self.assertIn("index differs", "\n".join(r.limitations))

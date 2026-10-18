import json
from pathlib import Path
import os
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import nullcheck
from nullcheck import core
from nullcheck.categorical import Codec, canonical
from tests.test_nullcheck import HAVE_BT, in_root
from fixtures import v02_features as F

FAST = nullcheck.Settings(n_cutoffs=12, reduce_budget=0)


class CodecTests(unittest.TestCase):
    def test_nullable_numeric_series_keep_numeric_comparison_contract(self):
        numeric = pd.Series([1., pd.NA], dtype="Float64")
        boolean = pd.Series([True, pd.NA], dtype="boolean")
        for name, values in (("numeric", numeric), ("boolean", boolean)):
            converted = Codec().convert(name, values)
            self.assertIs(converted, values)
            cats = []
            channels, problems, _ = core.normalise({name:converted}, values.index, cats)
            self.assertEqual(cats, [])
            if name == "numeric":
                self.assertEqual(channels[name][0], 1.)
                self.assertTrue(np.isnan(channels[name][1]))
            else:
                self.assertIn("boolean channel with missing values", problems[name])

    def test_normalisation_owns_numeric_and_categorical_arrays(self):
        for values in (np.arange(3.), Codec().convert("x", ["a", "b", "c"])):
            original = values.copy()
            channels, _, _ = core.normalise({"x":values}, pd.RangeIndex(3))
            values[:] = 0
            np.testing.assert_array_equal(channels["x"], original)

    def test_exact_equality_is_independent_of_numeric_tolerance(self):
        codec = Codec()
        a = codec.convert("x", ["up", "down"])
        b = codec.convert("x", ["down", "up"])
        result = core.compare_channel("x", a, b, 2, nullcheck.Settings(atol=1e20, rtol=1e20), categorical=True)
        self.assertEqual(len(result["diffs"]), 2)

    def test_mixed_state_values_are_not_coerced_to_strings(self):
        codec = Codec()
        a = codec.convert("x", [1, "label"])
        b = codec.convert("x", ["1", "label"])
        self.assertNotEqual(a[0], b[0])
        self.assertEqual(a[1], b[1])

    def test_missing_and_timestamp_canonical_values(self):
        codec = Codec()
        value = codec.convert("x", np.array([None, pd.NA, pd.NaT, np.datetime64("NaT", "ns"), "NaT"], dtype=object))
        self.assertTrue(np.isnan(value[:4]).all())
        self.assertTrue(np.isfinite(value[4]))
        self.assertEqual(canonical(pd.Timestamp("2020-01-01")), canonical(np.datetime64("2020-01-01")))
        self.assertNotEqual(canonical(pd.Timestamp("2020-01-01", tz="UTC")), canonical(pd.Timestamp("2020-01-01")))
        self.assertEqual(canonical(pd.Timestamp("2020-01-01", tz="UTC")), canonical(pd.Timestamp("2019-12-31 19:00", tz="America/New_York")))
        self.assertNotEqual(canonical("1"), canonical(1))
        self.assertNotEqual(canonical(b"1"), canonical("1"))
        self.assertEqual(canonical(1), canonical(1.))

    def test_failed_second_baseline_does_not_leave_categorical_contract(self):
        calls = []
        def callback(rows):
            calls.append(1)
            if len(calls) == 2:
                raise ValueError("second run failed")
            return {"x": Codec().convert("x", ["up"] * len(rows))}
        result = core.check(callback, F.prices(20), FAST)
        self.assertEqual(result.status, nullcheck.NOT_ASSESSED)
        self.assertEqual(result.categorical_channels, [])

    def test_collision_cannot_silently_compare_equal(self):
        with patch("nullcheck.categorical.fingerprint", return_value=1):
            codec = Codec()
            codec.convert("x", ["first"])
            with self.assertRaisesRegex(ValueError, "collision"):
                codec.convert("x", ["second"])

    def test_unsupported_object_does_not_call_repr(self):
        class Opaque:
            def __repr__(self): raise AssertionError("repr must not be called")
        with self.assertRaises(TypeError):
            Codec().convert("x", np.array(["label", Opaque()], dtype=object))


@unittest.skipUnless(HAVE_BT, "backtesting not installed")
class CategoricalRegressions(unittest.TestCase):
    def test_all_disclosed_categorical_leak_patterns_diverge(self):
        from fixtures import v024_bt as B
        raw = F.prices(100)
        for cls in (B.RegimeStr, B.PeakDate, B.LabelRows, B.LabelCategory,
                    B.SplitDate, B.NoneSentinel, B.NumericToNone, B.MixedFrameLeak, B.DatetimeRows, B.StringMatrix):
            with self.subTest(strategy=cls.__name__):
                r = nullcheck.check_backtesting(cls, raw, settings=FAST)
                self.assertEqual(r.status, nullcheck.DIVERGENCE, (r.reason, r.limitations))
                self.assertTrue(r.categorical_channels or cls in (B.NumericToNone, B.NoneSentinel))
        r = nullcheck.check_backtesting(B.Base, raw, prepare=B.prepare_label, settings=FAST)
        self.assertEqual(r.status, nullcheck.DIVERGENCE)
        self.assertIn("data:target", r.categorical_channels)

    def test_duplicate_dataframe_channels_cannot_hide_categorical_findings(self):
        from fixtures.v024_bt import Base, DuplicateFrame, StringifiedDuplicateFrame, prepare_duplicate
        for strategy in (DuplicateFrame, StringifiedDuplicateFrame):
            r = nullcheck.check_backtesting(strategy, F.prices(80), settings=FAST)
            self.assertEqual(r.status, nullcheck.NOT_ASSESSED)
            self.assertIn("attr:frame", r.channels)
            self.assertTrue(any("duplicate column names" in line for line in r.limitations))
        r = nullcheck.check_backtesting(Base, F.prices(80), prepare=prepare_duplicate, settings=FAST)
        self.assertEqual(r.status, nullcheck.NOT_ASSESSED)
        self.assertIn("duplicate column names", r.reason)

    def test_variable_length_future_dates_diverge_by_state_shape(self):
        from fixtures.v024_bt import FutureDates
        r = nullcheck.check_backtesting(FutureDates, F.prices(100), settings=FAST)
        self.assertEqual(r.status, nullcheck.DIVERGENCE)
        self.assertTrue(any(c.startswith("attr:dates") for c in r.channels))

    def test_legacy_schema_03_validates_but_needs_original_runner(self):
        from fixtures.v024_bt import Base
        with tempfile.TemporaryDirectory() as t:
            out = os.path.join(t, "record")
            nullcheck.check_backtesting(Base, F.prices(80), settings=FAST, out=out, root=t)
            path = Path(out)/"replay.json"
            rec = json.loads(path.read_text())
            rec["schema_version"] = "0.3.0"
            rec["check"]["version"] = rec["check"]["runner_version"] = "0.2.3"
            rec["contract"].pop("categorical_channels")
            rec["contract"].pop("categorical_encoding")
            path.write_text(json.dumps(rec))
            self.assertEqual(nullcheck.validate(out, t), [])
            self.assertEqual(nullcheck.replay(out, root=t)["outcome"], "UNSUPPORTED_RUNNER")

    def test_constant_metadata_compared_beside_causal_signal(self):
        from fixtures.v024_bt import ConstantMetadata
        r = nullcheck.check_backtesting(ConstantMetadata, F.prices(100), settings=FAST)
        self.assertEqual(r.status, nullcheck.NO_DIVERGENCE, r.reason)
        self.assertIn("attr:mode", r.categorical_channels)
        self.assertIn("attr:names[0]", r.categorical_channels)
        self.assertTrue(any(x["channel"] == "attr:mode" for p in r.probes for x in p["compared_ranges"]))
        self.assertTrue(any(x["channel"] == "attr:missing" for p in r.probes for x in p["skipped_ranges"]))

    def test_dependency_probe_encoding_switch_cannot_certify_numeric_coverage(self):
        from fixtures.v024_bt import EncodingSwitch
        raw = F.prices(80)
        raw["Close"] = np.linspace(150., 50., len(raw))
        raw["Open"] = raw.Close
        raw["High"] = raw.Close * 1.01
        raw["Low"] = raw.Close * .99
        r = nullcheck.check_backtesting(EncodingSwitch, raw, settings=FAST)
        self.assertEqual(r.status, nullcheck.NOT_ASSESSED)
        self.assertEqual(r.reason, "insufficient meaningful coverage")
        self.assertTrue(any("encoding changed" in line for line in r.limitations))

    def test_categorical_fingerprints_alone_do_not_qualify_meaningful_coverage(self):
        from fixtures.v024_bt import CategoricalOnly
        r = nullcheck.check_backtesting(CategoricalOnly, F.prices(100), settings=FAST)
        self.assertEqual(r.status, nullcheck.NOT_ASSESSED)
        self.assertEqual(r.reason, "insufficient meaningful coverage")

    def test_records_replay_categorical_findings_and_reduced_case(self):
        for name, prepare_name in (("SplitDate", None), ("FutureDates", None), ("Base", "prepare_label"), ("ConstantMetadata", None)):
            with self.subTest(strategy=name), tempfile.TemporaryDirectory() as t:
                strategy = in_root(t, "v024_bt", name)
                prepare = in_root(t, "v024_bt", prepare_name) if prepare_name else None
                out = os.path.join(t, "record")
                r = nullcheck.check_backtesting(strategy, F.prices(80), prepare=prepare,
                    settings=nullcheck.Settings(n_cutoffs=6, reduce_budget=60), out=out, root=t)
                self.assertEqual(nullcheck.validate(out, t), [])
                replay = nullcheck.replay(out, root=t)
                self.assertEqual(replay["outcome"], "REPRODUCED", replay)
                if r.reducer.get("reproducer"):
                    self.assertEqual(replay["small_reproducer"], "REPRODUCED", replay)
                record = json.loads((Path(out)/"replay.json").read_text())
                self.assertEqual(record["schema_version"], "0.4.0")
                self.assertEqual(record["contract"]["categorical_channels"], r.categorical_channels)

    def test_large_tolerance_record_uses_exact_equality_and_validates(self):
        from fixtures.v024_bt import SplitDate
        with tempfile.TemporaryDirectory() as t:
            out = os.path.join(t, "record")
            r = nullcheck.check_backtesting(SplitDate, F.prices(80),
                settings=nullcheck.Settings(n_cutoffs=3, reduce_budget=0, atol=1e20, rtol=1e20), out=out, root=t)
            self.assertEqual(r.status, nullcheck.DIVERGENCE)
            self.assertEqual(nullcheck.validate(out, t), [])
            rec = json.loads((Path(out)/"replay.json").read_text())
            rec["contract"]["categorical_channels"] = []
            from nullcheck.replay import validate_record
            self.assertTrue(any("tolerance" in e for e in validate_record(rec)))

    def test_collision_marks_attribute_unassessed_beside_causal_channel(self):
        from fixtures.v024_bt import ConstantMetadata
        with patch("nullcheck.categorical.fingerprint", return_value=1):
            r = nullcheck.check_backtesting(ConstantMetadata, F.prices(80), settings=FAST)
        self.assertEqual(r.status, nullcheck.NOT_ASSESSED)
        self.assertIn("collision", "\n".join(r.limitations))

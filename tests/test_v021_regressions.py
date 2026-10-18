import contextlib
import io
import json
import os
import tempfile
import unittest
import numpy as np
import pandas as pd
import nullcheck
from nullcheck import core
from nullcheck.__main__ import main
from fixtures import v02_features as F
from fixtures import v021_features as T
from tests.test_nullcheck import HAVE_BT, in_root


class ChronologyRegressions(unittest.TestCase):
    def test_RangeIndex_temporal_column_refuses_leak_and_clean_before_callback(self):
        raw = F.prices(300).iloc[::-1].reset_index()
        for fn in (T.lag1, T.sorted_sma):
            r = nullcheck.check(fn, raw)
            self.assertEqual(r.status, nullcheck.NOT_ASSESSED)
            self.assertEqual(r.runs_used, 0)
            self.assertIn("Temporal column 'Date'", r.reason)

    def test_temporal_column_is_checked_even_with_valid_datetime_index(self):
        raw = F.prices(100)
        raw["event_time"] = raw.index[::-1]
        r = nullcheck.check(T.identity, raw)
        self.assertEqual(r.status, nullcheck.NOT_ASSESSED)
        self.assertEqual(r.runs_used, 0)

    def test_period_columns_duplicates_NaT_and_shuffle_refused(self):
        raw = F.prices(100).reset_index()
        variants = []
        p = raw.copy(); p["Date"] = pd.period_range("2020-01-01", periods=len(p), freq="D")[::-1]; variants.append(p)
        p = raw.copy(); p.loc[1,"Date"] = p.loc[0,"Date"]; variants.append(p)
        p = raw.copy(); p.loc[1,"Date"] = pd.NaT; variants.append(p)
        p = raw.copy(); p["Date"] = p.Date.sample(frac=1, random_state=3).to_numpy(); variants.append(p)
        for p in variants:
            self.assertEqual(nullcheck.check(T.identity, p).runs_used, 0)
            self.assertEqual(nullcheck.check(T.identity, p).status, nullcheck.NOT_ASSESSED)

    def test_ISO_date_strings_named_and_custom_header_are_checked(self):
        raw = F.prices(100).iloc[::-1].reset_index()
        raw["Date"] = raw.Date.astype(str)
        for d in (raw, raw.rename(columns={"Date":"observed_at"})):
            r = nullcheck.check(T.identity, d)
            self.assertEqual(r.status, nullcheck.NOT_ASSESSED)
            self.assertEqual(r.runs_used, 0)

    def test_non_temporal_index_discloses_unverified_index_chronology(self):
        raw = F.prices(100).reset_index(drop=True)
        r = nullcheck.check(T.identity, raw)
        self.assertEqual(r.status, nullcheck.NOT_ASSESSED)
        self.assertTrue(any("index chronology not verified" in x for x in r.limitations))
        raw["Date"] = pd.date_range("2020-01-01", periods=len(raw))
        self.assertEqual(nullcheck.check(T.identity, raw).status, nullcheck.NOT_ASSESSED)

    def test_CLI_no_dates_csv_row_number_column_refused(self):
        with tempfile.TemporaryDirectory() as t:
            raw = F.prices(100).iloc[::-1].reset_index()
            data = os.path.join(t,"raw.csv"); raw.to_csv(data)
            in_root(t,"v021_features","lag1")
            for name in ("lag1","sorted_sma"):
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                    code = main(["check",os.path.join(t,"code/v021_features.py")+":"+name,"--data",data,"--no-dates"])
                self.assertEqual(code,3,buf.getvalue())
                self.assertIn("Temporal column 'Date'",buf.getvalue())

    def test_temporal_column_records_roundtrip_and_replay(self):
        for dtype in ("datetime", "period", "datetime_tz"):
            for reverse in (False,True):
                with self.subTest(dtype=dtype,reverse=reverse), tempfile.TemporaryDirectory() as t:
                    raw = F.prices(100).reset_index()
                    if dtype=="period": raw["Date"]=pd.period_range("2020-01-01",periods=len(raw),freq="D")
                    if dtype=="datetime_tz": raw["Date"]=pd.date_range("2020-01-01",periods=len(raw),tz="America/New_York")
                    if reverse: raw["Date"]=raw.Date.iloc[::-1].to_numpy()
                    fn = in_root(t,"v021_features","identity")
                    out = os.path.join(t,"record")
                    r = nullcheck.check(fn,raw,out=out,root=t)
                    self.assertEqual(r.status,nullcheck.NOT_ASSESSED)
                    self.assertEqual(nullcheck.validate(out,t),[])
                    self.assertEqual(nullcheck.replay(out,root=t)["outcome"],"REPRODUCED")


@unittest.skipUnless(HAVE_BT,"backtesting not installed")
class SnapshotRegressions(unittest.TestCase):
    def test_review_numeric_state_forms_flag(self):
        from fixtures import v021_bt as B
        for st in (B.NdarrayLevels,B.SeriesQuantiles,B.NpBoolScalar,B.FeatureMatrix,B.FloatScalarOK):
            with self.subTest(st=st.__name__):
                r=nullcheck.check_backtesting(st,B.prices())
                self.assertEqual(r.status,nullcheck.DIVERGENCE,(st,r.reason))
        r=nullcheck.check_backtesting(B.NpBoolScalar,B.prices())
        self.assertIn("attr:bull",r.boolean_channels)

    def test_state_vector_shape_does_not_switch_at_short_prefix(self):
        from fixtures import v021_bt as B
        from nullcheck.adapters.backtesting_py import make_feature_fn
        raw=B.prices()
        fn=make_feature_fn(B.ConstantStates,None,{}, {})
        full=fn(raw)
        for n in (1,2,3,4):
            prefix=fn(raw.iloc[:n])
            self.assertEqual(set(prefix),set(full))
            for c in full:
                np.testing.assert_array_equal(prefix[c],full[c][:n])
        self.assertEqual(nullcheck.check_backtesting(B.ConstantStates,raw).status,nullcheck.NOT_ASSESSED)

    def test_matrix_orientation_remains_stable_when_prefix_is_square(self):
        from fixtures import v021_bt as B
        from nullcheck.adapters.backtesting_py import make_feature_fn
        raw=B.prices(); fn=make_feature_fn(B.CausalMatrix,None,{}, {})
        full=fn(raw); pref=fn(raw.iloc[:2])
        for key in ("attr:rows","attr:last_axis"):
            np.testing.assert_array_equal(pref[key],full[key][:,:2])
        self.assertEqual(nullcheck.check_backtesting(B.CausalMatrix,raw).status,nullcheck.NO_DIVERGENCE)

    def test_dict_and_other_unsupported_attributes_are_named_and_unassessed(self):
        from fixtures import v021_bt as B
        r=nullcheck.check_backtesting(B.DictState,B.prices())
        self.assertEqual(r.status,nullcheck.NOT_ASSESSED)
        self.assertTrue(any("'lv'" in x and "builtins.dict" in x for x in r.limitations))
        r=nullcheck.check_backtesting(B.UnsupportedState,B.prices())
        self.assertEqual(r.status,nullcheck.NOT_ASSESSED)
        for name,typ in (("metadata","dict"),("_opaque","object")):
            self.assertTrue(any(repr(name) in x and "builtins."+typ in x for x in r.limitations),(name,r.limitations))
        self.assertIn("attr:description", r.categorical_channels)
        for framework in ("_broker","_data","_params","_indicators"):
            self.assertTrue(any(repr(framework) in x for x in r.limitations))

    def test_reduced_state_case_reproduces_with_a_fresh_adapter(self):
        from fixtures import v021_bt as B
        raw=B.prices()
        for st in (B.NdarrayLevels,B.SeriesQuantiles,B.NpBoolScalar,B.FeatureMatrix):
            r=nullcheck.check_backtesting(st,raw)
            rp=r.reducer["reproducer"]
            if rp:
                tiny=raw.iloc[rp["head"]:rp["stop"]]
                independent=nullcheck.check_backtesting(st,tiny)
                self.assertEqual(independent.status,nullcheck.DIVERGENCE,(st.__name__,rp,independent.reason))

    def test_review_state_and_matrix_records_replay(self):
        from fixtures import v021_bt as B
        for name in ("NdarrayLevels","SeriesQuantiles","NpBoolScalar","FeatureMatrix","DictState","ConstantStates","CausalMatrix"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as t:
                st=in_root(t,"v021_bt",name)
                out=os.path.join(t,"record")
                nullcheck.check_backtesting(st,B.prices(),out=out,root=t)
                self.assertEqual(nullcheck.validate(out,t),[])
                self.assertEqual(nullcheck.replay(out,root=t)["outcome"],"REPRODUCED")

    def test_R4b_quantile_ndarray_beside_self_I_flags_and_replays(self):
        from fixtures import v021_bt as B
        with tempfile.TemporaryDirectory() as t:
            st=in_root(t,"v021_bt","QuantileWithIndicator")
            out=os.path.join(t,"record")
            r=nullcheck.check_backtesting(st,B.prices(),out=out,root=t)
            self.assertEqual(r.status,nullcheck.DIVERGENCE)
            self.assertTrue(any(d["channel"].startswith("attr:bounds[") for d in r.divergences))
            self.assertEqual(nullcheck.validate(out,t),[])
            self.assertEqual(nullcheck.replay(out,root=t)["outcome"],"REPRODUCED")

    def test_multidimensional_numeric_state_and_numpy_scalar_controls(self):
        from fixtures import v021_bt as B
        r=nullcheck.check_backtesting(B.TensorState,B.prices())
        self.assertEqual(r.status,nullcheck.DIVERGENCE)
        self.assertTrue(any(d["channel"].startswith("attr:tensor[") for d in r.divergences))
        r=nullcheck.check_backtesting(B.ConstantMultiShape,B.prices())
        self.assertEqual(r.status,nullcheck.NOT_ASSESSED,r.reason)
        self.assertEqual(r.reason,"insufficient meaningful coverage")
        self.assertEqual(len(r.channels),31)  # 24 tensor + 2 tuple + three scalars + two shapes

    def test_every_unsupported_type_is_named_without_losing_numeric_channels(self):
        from fixtures import v021_bt as B
        from nullcheck.adapters.backtesting_py import make_feature_fn
        raw=B.prices()
        snapshot=make_feature_fn(B.UnsupportedGallery,None,{},{})(raw)
        channels,problems,_=core.normalise(snapshot,raw.index)
        self.assertEqual(set(channels), {"attr:causal", "attr:numpy_text", "attr:numpy_date", "attr:zero_dim_date", "attr:timestamp", "attr:mixed_tuple[0]", "attr:mixed_tuple[1]", "attr:mixed_tuple[shape]"})
        names=("mapping","numpy_complex",
               "structured","numpy_void","nested","ragged_tuple","series_objects",
               "empty_tensor","empty_rows","empty_frame","failing_array")
        self.assertEqual(set(problems),{"attr:"+name for name in names})
        for compared in ("numpy_text", "numpy_date", "zero_dim_date", "timestamp"):
            self.assertIn("attr:"+compared, channels)
        for name in names:
            self.assertTrue(any(repr(name) in line and "not compared" in line for line in snapshot.limitations),name)
        for cls,status in ((B.UnsupportedGallery,nullcheck.NOT_ASSESSED),(B.UnsupportedWithLeak,nullcheck.DIVERGENCE)):
            r=nullcheck.check_backtesting(cls,raw)
            self.assertEqual(r.status,status,r.reason)
            for name in names:
                self.assertTrue(any(repr(name) in line and "not compared" in line for line in r.limitations),name)
            self.assertIn("attr:causal",r.channels)

    def test_unsupported_type_gallery_record_replays(self):
        from fixtures import v021_bt as B
        with tempfile.TemporaryDirectory() as t:
            st=in_root(t,"v021_bt","UnsupportedGallery")
            out=os.path.join(t,"record")
            r=nullcheck.check_backtesting(st,B.prices(),out=out,root=t)
            self.assertEqual(r.status,nullcheck.NOT_ASSESSED)
            self.assertEqual(nullcheck.validate(out,t),[])
            self.assertEqual(nullcheck.replay(out,root=t)["outcome"],"REPRODUCED")

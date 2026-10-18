import os
import tempfile
import unittest
import numpy as np
import pandas as pd
import nullcheck
from nullcheck import core
from fixtures import v02_features as D, v024_scope as F
from tests.test_nullcheck import HAVE_BT, in_root

FAST = nullcheck.Settings(n_cutoffs=12, reduce_budget=0)

def prices(n=180):
    raw=D.prices(n)
    raw["Open"] = raw.Close * (1 + .005*np.sin(np.arange(n)))
    raw["High"] = raw.Close * 1.04
    raw["Low"] = raw.Close * .99
    return raw

class RowLocalCoverage(unittest.TestCase):
    def test_disclosed_row_local_transforms_are_not_meaningful(self):
        for gaps in (False, True):
            raw=prices()
            if gaps: raw.iloc[[40,41,100], raw.columns.get_loc("Volume")] = np.nan
            before=raw.copy(deep=True)
            for fn in (F.log_close,F.intrabar_return,F.typical_price,F.row_event,F.price_volume):
                with self.subTest(fn=fn.__name__,gaps=gaps):
                    r=nullcheck.check(fn,raw,settings=FAST)
                    self.assertEqual(r.status,nullcheck.NOT_ASSESSED,r.reason)
                    self.assertEqual(r.reason,"insufficient meaningful coverage")
                    self.assertTrue(any("row-local" in line for line in r.limitations))
                    self.assertEqual(r.runs_used,4+2*len(r.probes))
                    self.assertTrue(raw.equals(before))

    def test_cross_row_controls_keep_meaningful_coverage(self):
        for fn in (F.sma,F.ema,F.change,F.lag,F.cummax,F.rolling_z,F.mixed):
            with self.subTest(fn=fn.__name__):
                r=nullcheck.check(fn,prices(),settings=FAST)
                self.assertEqual(r.status,nullcheck.NO_DIVERGENCE,(r.reason,r.limitations))
                self.assertEqual(r.runs_used,4+2*len(r.probes))

    def test_divergence_wins_even_when_transform_commutes_with_reversal(self):
        for fn in (F.future,F.global_z):
            r=nullcheck.check(fn,prices(),settings=FAST)
            self.assertEqual(r.status,nullcheck.DIVERGENCE)
            self.assertEqual(r.runs_used,2+2*len(r.probes))

    def test_row_local_and_cross_row_records_replay(self):
        for name in ("log_close","row_event","sma","mixed"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as t:
                fn=in_root(t,"v024_scope",name)
                out=os.path.join(t,"record")
                nullcheck.check(fn,prices(),settings=FAST,out=out,root=t)
                self.assertEqual(nullcheck.validate(out,t),[])
                self.assertEqual(nullcheck.replay(out,root=t)["outcome"],"REPRODUCED")

class ExplicitTemporalNames(unittest.TestCase):
    def test_substrings_are_not_time_keys(self):
        raw=prices().reset_index(drop=True)
        for name in ("sentiment","candidate","validated","consolidated","liquidated","uptime","lifetime_value","timeframe"):
            with self.subTest(name=name):
                self.assertFalse(core.temporal_key(name))
                data=raw.assign(**{name:np.sin(np.arange(len(raw)))})
                r=nullcheck.check(F.sma,data,settings=FAST)
                self.assertEqual(r.status,nullcheck.NO_DIVERGENCE,r.reason)

    def test_explicit_epoch_and_datetime_names_still_refuse_newest_first(self):
        raw=prices().reset_index(drop=True)
        for name in ("Date","trade_date","eventTimestamp","DateTime","timestamp_ms","epochSeconds","ts","unixms","unix_ns"):
            with self.subTest(name=name):
                self.assertTrue(core.temporal_key(name))
                raw[name]=1609459200+np.arange(len(raw))[::-1]*86400
                r=nullcheck.check(F.sma,raw,settings=FAST)
                self.assertEqual(r.status,nullcheck.NOT_ASSESSED)
                self.assertEqual(r.runs_used,0)
                raw=raw.drop(columns=[name])

    def test_sentiment_is_perturbed_as_numeric_data(self):
        raw=prices().reset_index(drop=True)
        raw["sentiment"]=np.sin(np.arange(len(raw)))
        r=nullcheck.check(lambda d:d.sentiment.rolling(5).mean(),raw,settings=FAST)
        self.assertEqual(r.status,nullcheck.NO_DIVERGENCE,r.reason)

class InertObjectCoverage(unittest.TestCase):
    def test_discarded_literal_noop_is_not_a_data_input(self):
        r=nullcheck.check(F.progress_only,prices(),settings=FAST)
        self.assertEqual(r.status,nullcheck.NO_DIVERGENCE,r.reason)
        self.assertFalse(any(c.startswith("input:") for c in r.channels))

    def test_data_methods_attributes_and_mutators_remain_inputs(self):
        for fn in (F.object_data,F.object_attribute,F.object_mutation):
            with self.subTest(fn=fn.__name__):
                r=nullcheck.check(fn,prices(),settings=FAST)
                self.assertTrue(any(c.startswith("input:object:") for c in r.channels))
                self.assertNotEqual(r.status,nullcheck.NO_DIVERGENCE)

    def test_inert_progress_record_replays(self):
        with tempfile.TemporaryDirectory() as t:
            fn=in_root(t,"v024_scope","progress_only")
            out=os.path.join(t,"record")
            nullcheck.check(fn,prices(),settings=FAST,out=out,root=t)
            self.assertEqual(nullcheck.validate(out,t),[])
            self.assertEqual(nullcheck.replay(out,root=t)["outcome"],"REPRODUCED")

@unittest.skipUnless(HAVE_BT,"backtesting not installed")
class AdapterScope(unittest.TestCase):
    def test_row_local_strategy_is_insufficient(self):
        from fixtures.v024_scope_bt import RowLocal
        r=nullcheck.check_backtesting(RowLocal,prices(),settings=FAST)
        self.assertEqual(r.status,nullcheck.NOT_ASSESSED)
        self.assertEqual(r.reason,"insufficient meaningful coverage")

    def test_empty_lists_and_tuples_are_excluded_only_while_empty(self):
        from fixtures.v024_scope_bt import EmptyList, EmptyThenPopulated, EmptyOnly
        r=nullcheck.check_backtesting(EmptyList,prices(),settings=FAST)
        self.assertEqual(r.status,nullcheck.NO_DIVERGENCE,r.reason)
        self.assertNotIn("attr:history",r.channels)
        self.assertTrue(any("history" in line and "EXCLUDED empty" in line for line in r.limitations))
        r=nullcheck.check_backtesting(EmptyThenPopulated,prices(),settings=FAST)
        self.assertEqual(r.status,nullcheck.NOT_ASSESSED)
        self.assertTrue(any("empty metadata container changed" in line for line in r.limitations))
        r=nullcheck.check_backtesting(EmptyOnly,prices(),settings=FAST)
        self.assertEqual(r.status,nullcheck.NOT_ASSESSED)

    def test_empty_list_record_replays(self):
        with tempfile.TemporaryDirectory() as t:
            strategy=in_root(t,"v024_scope_bt","EmptyList")
            out=os.path.join(t,"record")
            nullcheck.check_backtesting(strategy,prices(),settings=FAST,out=out,root=t)
            self.assertEqual(nullcheck.validate(out,t),[])
            self.assertEqual(nullcheck.replay(out,root=t)["outcome"],"REPRODUCED")

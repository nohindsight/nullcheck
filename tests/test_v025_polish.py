import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import pandas as pd
import nullcheck
from nullcheck import core
from nullcheck.presentation import token_display
from nullcheck.categorical import canonical
from fixtures import v025 as F, v02_features as D
from tests.test_nullcheck import HAVE_BT, in_root

FAST = nullcheck.Settings(n_cutoffs=6, reduce_budget=0)


class DefaultStateLabels(unittest.TestCase):
    def test_mutable_positional_and_keyword_defaults_are_state_not_external(self):
        for fn in (F.default_columns, F.kwdefault_columns, F.default_predictions):
            with self.subTest(fn=fn.__name__):
                result = nullcheck.check(fn, D.prices(80), settings=FAST)
                self.assertEqual(result.status, nullcheck.NOT_ASSESSED)
                self.assertTrue(any(c.startswith('input:default:') for c in result.channels))
                rendered = nullcheck.render(result)
                self.assertIn('Default-argument state not assessed', rendered)
                self.assertNotIn('External precomputed inputs', rendered)
                self.assertFalse(result.raw_prefix_recomputed)

    def test_genuine_global_data_keeps_external_label(self):
        result = nullcheck.check(F.global_input, D.prices(80), settings=FAST)
        self.assertEqual(result.status, nullcheck.NOT_ASSESSED)
        self.assertIn('External precomputed inputs', nullcheck.render(result))

    def test_default_record_validates_without_claiming_replayable_state(self):
        with tempfile.TemporaryDirectory() as t:
            fn = in_root(t, 'v025', 'default_columns')
            out = os.path.join(t, 'record')
            nullcheck.check(fn, D.prices(80), settings=FAST, out=out, root=t)
            self.assertEqual(nullcheck.validate(out, t), [])
            rec = json.loads((Path(out) / 'replay.json').read_text())
            self.assertIn('default-argument state', rec['reproduction']['reason'])
            self.assertNotIn('external', rec['reproduction']['reason'])
            self.assertEqual(nullcheck.replay(out, root=t)['outcome'], 'UNAVAILABLE')


class DisplayTokens(unittest.TestCase):
    def test_safe_strings_bytes_numbers_and_dates(self):
        self.assertEqual(token_display(canonical('x\n\x1b[31m')), '"x\\n\\u001b[31m"')
        self.assertEqual(token_display(canonical(b'x\n')), "b'x\\n'")
        self.assertEqual(token_display(canonical(1.25)), '1.25')
        self.assertEqual(token_display(canonical(True)), 'True')
        self.assertEqual(token_display(canonical(pd.Timestamp('2020-01-02', tz='UTC'))), '2020-01-02T00:00:00+00:00')
        self.assertEqual(token_display(canonical(pd.Timestamp('2020-01-02').date())), '2020-01-02')

    def test_readme_and_coverage_disclose_all_named_limits(self):
        for name in ('README.md', 'COVERAGE.md'):
            text = (Path(__file__).resolve().parents[1] / name).read_text()
            for marker in ('K4r', 'K5', 'K8', 'K10', 'K11', 'AVAILABLE', 'not that it was traded'):
                self.assertIn(marker, text, name)


@unittest.skipUnless(HAVE_BT, 'backtesting not installed')
class FindingPresentation(unittest.TestCase):
    def test_strings_and_categories_are_readable_with_row_and_prefix(self):
        from fixtures.v025_bt import StringRows, CategoryRows, MissingString
        for cls in (StringRows, CategoryRows, MissingString):
            result = nullcheck.check_backtesting(cls, D.prices(80), settings=FAST)
            self.assertEqual(result.status, nullcheck.DIVERGENCE)
            line = next(x for x in nullcheck.render(result).splitlines() if x.startswith('  DIVERGENCE'))
            self.assertIn('row 0', line)
            self.assertIn('prefix length 1', line)
            self.assertIn('full run "long"', line)
            self.assertIn('prefix run ' + ('missing' if cls is MissingString else '"short"'), line)
            self.assertNotIn(str(result.findings[0]['full_value']), line)

    def test_timestamp_and_bookkeeping_hint_with_available_not_traded_meaning(self):
        from fixtures.v025_bt import LastDay
        raw = D.prices(80)
        result = nullcheck.check_backtesting(LastDay, raw, settings=FAST)
        text = nullcheck.render(result)
        self.assertIn(raw.index[-1].isoformat(), text)
        self.assertIn(raw.index[0].isoformat(), text)
        self.assertIn('data-end bookkeeping?', text)
        self.assertIn('AVAILABLE', text)
        self.assertIn('does not establish that it was traded', text)

    def test_numeric_and_boolean_signals_headline_even_when_metadata_is_earlier(self):
        from fixtures.v025_bt import NumericAndMetadata, SignalAndMetadata
        for cls, name in ((NumericAndMetadata, 'I[1]:zscore'), (SignalAndMetadata, 'I[1]:buy_signal')):
            result = nullcheck.check_backtesting(cls, D.prices(80), settings=FAST)
            lines = [x for x in nullcheck.render(result).splitlines() if x.startswith('  DIVERGENCE')]
            self.assertIn(name, lines[0])
            self.assertTrue(any('attr:lastday' in line for line in lines[1:]))
            # Historical comparison selection is unchanged, including its row-0 metadata tie.
            self.assertEqual(result.probes[0]['first_divergence']['channel'],
                             'attr:lastday' if cls is NumericAndMetadata else name)
            self.assertEqual({f['channel'] for f in result.findings}, {'attr:lastday', name})

    def test_display_collection_does_not_change_detection_or_runs(self):
        from fixtures.v025_bt import NumericAndMetadata
        raw = D.prices(80)
        result = nullcheck.check_backtesting(NumericAndMetadata, raw, settings=FAST)
        with patch('nullcheck.presentation.finding', return_value={}):
            without = nullcheck.check_backtesting(NumericAndMetadata, raw, settings=FAST)
        for key in ('status', 'reason', 'probes', 'channels', 'reducer', 'runs_used', 'max_runs', 'categorical_channels'):
            self.assertEqual(getattr(result, key), getattr(without, key), key)

    def test_record_keeps_secondary_fingerprints_and_replays(self):
        with tempfile.TemporaryDirectory() as t:
            cls = in_root(t, 'v025_bt', 'NumericAndMetadata')
            out = os.path.join(t, 'record')
            result = nullcheck.check_backtesting(cls, D.prices(80), settings=FAST, out=out, root=t)
            self.assertEqual(nullcheck.validate(out, t), [])
            self.assertEqual(nullcheck.replay(out, root=t)['outcome'], 'REPRODUCED')
            supplement = json.loads((Path(out) / 'findings.json').read_text())
            self.assertEqual(supplement['findings'], result.findings)
            self.assertEqual({f['channel'] for f in supplement['findings']}, {'attr:lastday', 'I[1]:zscore'})
            stamp = next(f for f in supplement['findings'] if f['channel'] == 'attr:lastday')
            self.assertIsInstance(stamp['full_value'], float)
            self.assertEqual(stamp['full_display'], D.prices(80).index[-1].isoformat())
            rec = json.loads((Path(out) / 'replay.json').read_text())
            self.assertEqual(rec['probes'][0]['first_divergence']['full_value'], stamp['full_value'])

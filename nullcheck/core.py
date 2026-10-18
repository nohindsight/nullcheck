"""LEAK check: indicator prefix invariance (check id 'indicator_prefix_invariance').

Contract (REPLAY_FORMAT.md, schema 0.4.0): for a prefix length L the callback receives a
fresh copy of raw rows [0, L). For every channel and every eligible row i (0 <= i < L - terminal_excluded):
  - both values finite: divergence if |a-b| > atol + rtol*max(|a|,|b|)            (kind VALUE)
  - exactly one value finite: divergence                                           (kind MISSINGNESS)
  - matching NaN: skipped with a reason (never counted as numeric coverage)
  - other non-finite pairs: unassessed, without hiding findings in other cells
Rows are positions in the original raw input. Repeat runs (baseline and every prefix) must agree exactly, or the
affected output is NOT_ASSESSED. NO_DIVERGENCE_OBSERVED means only: no difference at these finite probes.
"""
from __future__ import annotations

import dataclasses
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .categorical import CategoricalValues

NO_DIVERGENCE = "NO_DIVERGENCE_OBSERVED"
DIVERGENCE = "DIVERGENCE_OBSERVED"
NOT_ASSESSED = "NOT_ASSESSED"
ERROR = "ERROR"

MAX_PROBES = 128
R_DECLARED = "Declared warm-up rows."
R_INFERRED = "Inferred warm-up: leading rows NaN in both runs (not proof of causal availability)."
R_TERMINAL = "Declared terminal exclusion."
R_PROBE_NA = "Probe not assessed."
R_MATCHED_NAN = "Matching NaN: sparse signal or missing output; no numeric coverage."
R_INVALID = "Invalid numeric cells: non-finite pair or magnitude above 1e300."
R_CHANNEL_NA = "Channel not assessed: "
R_MEANINGFUL = "insufficient meaningful coverage"
MAX_MAGNITUDE = 1e300  # larger finite values are not compared (overflow-safe tolerance arithmetic)

FeatureFn = Callable[[pd.DataFrame], Any]


class FeatureSnapshot(dict):
    """Internal adapter output with explicit coverage notes; numeric values remain ordinary channels."""
    def __init__(self):
        super().__init__()
        self.limitations = []
        self.unassessed = {}


@dataclasses.dataclass(frozen=True)
class Settings:
    warmup_rows: int = 0
    terminal_excluded_rows: int = 0
    n_cutoffs: int = 24
    seed: int = 0
    rtol: float = 1e-9
    atol: float = 1e-12
    nan_cutoffs: bool = True
    reduce_budget: int = 120  # callback runs the reducer may spend, including its final verification

    def validate(self) -> None:
        for name in ("warmup_rows", "terminal_excluded_rows", "n_cutoffs", "seed", "reduce_budget"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int) or v < 0:
                raise ValueError(f"{name} must be a non-negative integer, got {v!r}")
        for name in ("rtol", "atol"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) or v < 0:
                raise ValueError(f"{name} must be a finite non-negative number, got {v!r}")
        if self.n_cutoffs < 1:
            raise ValueError("n_cutoffs must be >= 1")


# ---- output normalisation ----------------------------------------------------------------------------------------
def normalise(out: Any, rows_index: pd.Index, categorical_channels=None) -> Tuple[Dict[str, np.ndarray], Dict[str, str], List[str]]:
    """Callback output -> ({channel: float array aligned to rows}, {channel: problem}, [boolean channels]).
    Mapping: numeric -> float; bool -> 0.0/1.0 (channel listed as boolean); pandas missing markers (None, pd.NA, NaT in
    numeric/boolean extension dtypes) -> NaN. Generic strings/objects remain unsupported; the adapter supplies
    explicitly marked categorical fingerprints and records their channel names for exact equality.
    pandas outputs must carry the input index unchanged (same labels, same order)."""
    categorical_channels = categorical_channels if categorical_channels is not None else []
    n = len(rows_index)
    chans: Dict[str, np.ndarray] = {}
    bools: List[str] = []
    problems: Dict[str, str] = {}

    def add(name: str, value: Any) -> None:
        if isinstance(value, pd.DataFrame):
            if not value.index.equals(rows_index):
                problems[(name or "output")] = f"{name or 'output'}: DataFrame index differs from the input rows (reordered, dropped or added rows)"
                return
            for col in value.columns:
                add(f"{name}.{col}" if name else str(col), value[col])
            return
        if isinstance(value, pd.Series):
            if not value.index.equals(rows_index):
                problems[(name or "output")] = f"{name}: Series index differs from the input rows (reordered, dropped or added rows)"
                return
            dt = value.dtype
            if pd.api.types.is_bool_dtype(dt):
                if value.isna().any():
                    problems[(name or "output")] = f"{name}: boolean channel with missing values"
                    return
                bools.append(name)
                chans[name] = value.to_numpy(dtype=float)
                return
            if pd.api.types.is_complex_dtype(dt):
                problems[(name or "output")] = f"{name}: complex values are not supported (would lose the imaginary part)"
                return
            if pd.api.types.is_numeric_dtype(dt):
                chans[name] = value.to_numpy(dtype=float, na_value=np.nan)
                return
            problems[(name or "output")] = f"{name}: non-numeric dtype {dt}; only numeric and boolean channels are compared"
            return
        try:
            arr = np.asanyarray(value)
        except (ValueError, TypeError) as e:
            problems[name] = f"{name}: cannot form an array ({type(e).__name__})"
            return
        if arr.ndim == 2 and arr.shape[-1] == n:
            for j in range(arr.shape[0]):
                add(f"{name}[{j}]", arr[j])
            return
        if arr.ndim != 1 or arr.shape[0] != n:
            problems[(name or "output")] = f"{name}: shape {arr.shape} is not one value per input row ({n})"
            return
        if np.iscomplexobj(arr):
            problems[(name or "output")] = f"{name}: complex values are not supported (would lose the imaginary part)"
        elif arr.dtype == bool:
            bools.append(name)
            chans[name] = arr.astype(float)
        elif np.issubdtype(arr.dtype, np.number):
            chans[name] = np.array(arr, dtype=float, copy=True)
            if isinstance(arr, CategoricalValues):
                categorical_channels.append(name)
        else:
            problems[(name or "output")] = f"{name}: non-numeric dtype {arr.dtype}; only numeric and boolean channels are compared"

    if isinstance(out, FeatureSnapshot):
        problems.update(out.unassessed)
    if isinstance(out, dict):
        for k, v in out.items():
            add(str(k), v)
    elif isinstance(out, pd.DataFrame):
        add("", out)
    else:
        add("feature", out)
    if not chans and not problems:
        problems["(output)"] = "callback returned no channels"
    return chans, problems, bools


def identical(a: np.ndarray, b: np.ndarray) -> bool:
    """Exact repeatability: same shape and identical values, NaN matching NaN and inf matching inf of the same sign."""
    return a.shape == b.shape and bool(np.array_equal(a, b, equal_nan=True))


# ---- per-channel comparison --------------------------------------------------------------------------------------
def _runs(mask: np.ndarray, offset: int) -> List[Tuple[int, int]]:
    """[start, stop) runs where mask is True, shifted by offset."""
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            out.append((i + offset, j + offset)); i = j
        else:
            i += 1
    return out


def _state(x: float) -> str:
    if np.isfinite(x):
        return "FINITE"
    if np.isnan(x):
        return "NAN"
    return "POS_INF" if x > 0 else "NEG_INF"


def compare_channel(channel: str, full: np.ndarray, prefix: np.ndarray, L: int, s: Settings, *, categorical=False) -> dict:
    """Audit warm-up too. Matching NaNs are skipped, never counted as numeric agreement.
    All other valid cells are compared; invalid cells cannot erase a finding elsewhere."""
    hi = max(0, L - s.terminal_excluded_rows)
    a, b = full[:hi], prefix[:hi]
    fa, fb = np.isfinite(a), np.isfinite(b)
    matched_nan = np.isnan(a) & np.isnan(b)
    big = (fa & (np.abs(a) > MAX_MAGNITUDE)) | (fb & (np.abs(b) > MAX_MAGNITUDE))
    invalid = ((~fa & ~fb) & ~matched_nan) | big
    ok = ~(matched_nan | invalid)
    compared = [{"channel": channel, "start": st, "stop": sp} for st, sp in _runs(ok, 0)]
    skipped = []
    for st, sp in _runs(matched_nan, 0):
        skipped.append({"channel": channel, "start": st, "stop": sp,
                        "reason": R_INFERRED if st == 0 else R_MATCHED_NAN})
    for st, sp in _runs(invalid, 0):
        skipped.append({"channel": channel, "start": st, "stop": sp, "reason": R_INVALID})
    if hi < L:
        skipped.append({"channel": channel, "start": hi, "stop": L, "reason": R_TERMINAL})
    both = ok & fa & fb
    av, bv = np.where(both, a, 0.0), np.where(both, b, 0.0)
    with np.errstate(over="ignore", invalid="ignore"):
        value_div = both & ((av != bv) if categorical else (np.abs(av - bv) > s.atol + s.rtol * np.maximum(np.abs(av), np.abs(bv))))
    miss_div = ok & (fa ^ fb)
    diffs = [{"channel": channel, "row": int(k), "kind": "MISSINGNESS" if miss_div[k] else "VALUE",
              "full_value": float(a[k]) if fa[k] else None, "prefix_value": float(b[k]) if fb[k] else None,
              "full_state": _state(a[k]), "prefix_state": _state(b[k])}
             for k in np.flatnonzero(value_div | miss_div)]
    problem = None
    if invalid.any():
        k = int(np.flatnonzero(invalid)[0])
        problem = f"channel {channel!r} row {k}: magnitude above {MAX_MAGNITUDE:g} or non-finite in both runs other than matching NaN"
    return {"compared": compared, "skipped": skipped, "diffs": diffs, "assessed": int(ok.sum()),
            "unassessable": problem}


def meaningful_coverage(full: Dict[str, np.ndarray], raw: pd.DataFrame, probes: List[dict],
                        settings: Settings) -> Tuple[List[str], Dict[str, str]]:
    """Conservative minimum on *compared finite cells*, never on unseen tail values.

    Reject constants, raw-column/index copies, and affine row counters. A remaining
    varying output is only an observed candidate for meaningful coverage: value
    comparisons do not prove arbitrary program/data provenance or strategy use.
    External input and unsupported-channel checks remain independent gates.
    This screen does not execute the callback; a separate bounded dependency
    probe checks remaining candidates before they can qualify.
    """
    compared, sparse = {}, set()
    for probe in probes:
        sparse.update(r["channel"] for r in probe["skipped_ranges"] if r["reason"] == R_MATCHED_NAN)
        for region in probe["compared_ranges"]:
            name = region["channel"]
            if name in full:
                if name not in compared:
                    compared[name] = np.zeros(len(raw), dtype=bool)
                compared[name][region["start"]:region["stop"]] = True
    inputs = {}
    for name in raw.columns:
        col = raw[name]
        if pd.api.types.is_numeric_dtype(col.dtype) and not pd.api.types.is_complex_dtype(col.dtype):
            inputs[f"raw column {str(name)!r}"] = col.to_numpy(dtype=float, na_value=np.nan)
    if pd.api.types.is_numeric_dtype(raw.index.dtype):
        inputs["raw index"] = raw.index.to_numpy(dtype=float)

    def same(a, b):
        with np.errstate(over="ignore", invalid="ignore"):
            return bool(np.all(np.isfinite(a) & np.isfinite(b) &
                (np.abs(a - b) <= settings.atol + settings.rtol * np.maximum(np.abs(a), np.abs(b)))))

    eligible, excluded = [], {}
    for name, values in sorted(full.items()):
        mask = compared.get(name, np.zeros(len(raw), dtype=bool)) & np.isfinite(values)
        positions = np.flatnonzero(mask)
        v = values[mask]
        if len(v) < 2:
            excluded[name] = "fewer than two compared finite rows"
        elif same(v, v[0]) and name not in sparse:
            excluded[name] = "constant on compared finite rows (including broadcast init-state values)"
        else:
            copied = next((label for label, arr in inputs.items() if same(v, arr[mask])), None)
            if copied:
                excluded[name] = "pass-through of " + copied + " on compared finite rows"
                continue
            with np.errstate(over="ignore", invalid="ignore"):
                counter = v[0] + (v[-1] - v[0]) * ((positions - positions[0]) / (positions[-1] - positions[0]))
            if same(v, counter) and name not in sparse:
                excluded[name] = "pure affine row counter on compared finite rows"
            else:
                eligible.append(name)
    return eligible, excluded


# ---- cutoffs -----------------------------------------------------------------------------------------------------
def _even_sample(values, count):
    values = sorted(set(int(x) for x in values))
    if len(values) <= count:
        return values
    return [values[i] for i in np.linspace(0, len(values) - 1, count).round().astype(int)]


def cutoff_candidates(n: int, s: Settings, raw: Optional[pd.DataFrame] = None) -> dict:
    """Data-aware boundaries are probe priorities, not evidence of a leak.
    No callback output or held-out test case is used in planning."""
    hi = n - 1
    base = np.linspace(1, hi, num=min(MAX_PROBES, max(1, s.n_cutoffs - s.n_cutoffs // 2))).round().astype(int).tolist()
    base += np.random.default_rng(s.seed).integers(1, hi + 1, size=min(MAX_PROBES, s.n_cutoffs // 2)).tolist()
    groups = {"grid_and_seeded": sorted(set(base))}
    if raw is None or not s.nan_cutoffs:
        return groups

    def boundaries(mask):
        mask = np.asarray(mask, dtype=bool)
        # Start and end of every anomalous run; single-bar anomalies keep their exact cutoff.
        prior = np.r_[False, mask[:-1]]
        following = np.r_[mask[1:], False]
        return [int(i) + 1 for i in np.flatnonzero(mask & (~prior | ~following)) if i + 1 < n]

    groups["missing"] = boundaries(raw.isna().any(axis=1))
    numeric = raw.select_dtypes(include=[np.number])
    groups["zero"] = boundaries(numeric.eq(0).any(axis=1))
    price_cols = [c for c in ("Open", "High", "Low", "Close") if c in numeric]
    if price_cols:
        prices = numeric[price_cols]
        groups["stale"] = boundaries(prices.eq(prices.shift(1)).all(axis=1))
        # Large price discontinuities (robust median/MAD threshold) probe both sides.
        with np.errstate(divide="ignore", invalid="ignore"):
            changes = prices.pct_change(fill_method=None).abs().max(axis=1).replace([np.inf, -np.inf], np.nan)
        median = changes.median()
        mad = (changes - median).abs().median()
        jumps = np.flatnonzero((changes > median + 6 * max(float(mad), 1e-12)).to_numpy())
        groups["changepoint"] = [int(L) for i in jumps for L in (i, i + 1) if 1 <= L < n]
    if isinstance(raw.index, (pd.DatetimeIndex, pd.PeriodIndex)):
        months = np.asarray([(d.year, d.month) for d in raw.index])
        groups["month_end"] = (np.flatnonzero((months[1:] != months[:-1]).any(axis=1)) + 1).tolist()
    return groups


def choose_cutoffs(n: int, s: Settings, raw: Optional[pd.DataFrame] = None) -> List[int]:
    """Always include short prefixes to audit filled warm-up. At most 128 total.
    Reserve up to half the cap for the grid; distribute remaining slots across anomaly types.
    If a category exceeds the budget, evenly sample it across the entire history."""
    if n < 2:
        return []
    groups = cutoff_candidates(n, s, raw)
    events = {k: v for k, v in groups.items() if k != "grid_and_seeded" and v}
    chosen = set(_even_sample(groups["grid_and_seeded"], MAX_PROBES // 2 if events else MAX_PROBES))
    remaining = {k: set(v) - chosen for k, v in events.items()}
    while len(chosen) < MAX_PROBES and any(remaining.values()):
        active = [k for k in remaining if remaining[k]]
        quota = max(1, (MAX_PROBES - len(chosen)) // len(active))
        for k in active:
            chosen.update(_even_sample(remaining[k], min(quota, MAX_PROBES - len(chosen))))
            for vals in remaining.values():
                vals.difference_update(chosen)
            if len(chosen) == MAX_PROBES:
                break
    return sorted(chosen)


# ---- the check ---------------------------------------------------------------------------------------------------
class Runner:
    """Calls the user's function on fresh copies of raw row ranges and counts calls."""

    def __init__(self, fn: FeatureFn, raw: pd.DataFrame):
        self.fn, self.raw, self.runs = fn, raw, 0
        self.limitations = set()
        self.categorical_channels = []
        from .inputs import ExternalInputs
        self.external = ExternalInputs(fn)

    def run(self, start: int, stop: int) -> Tuple[Dict[str, np.ndarray], Dict[str, str], List[str]]:
        self.runs += 1
        rows = self.raw.iloc[start:stop].copy(deep=True)
        with self.external.observe():
            output = self.fn(rows)
        if isinstance(output, FeatureSnapshot):
            self.limitations.update(output.limitations)
        self.categorical_channels = []
        return normalise(output, rows.index, self.categorical_channels)

    def run_twice(self, start: int, stop: int) -> Tuple[Optional[Dict[str, np.ndarray]], str, List[str]]:
        """Keep stable numeric channels even if another channel is invalid or unstable."""
        self.excluded = {}
        try:
            c1, p1, b1 = self.run(start, stop)
            cat1 = set(self.categorical_channels)
            c2, p2, b2 = self.run(start, stop)
            cat2 = set(self.categorical_channels)
        except Exception as e:
            self.categorical_channels = []
            return None, f"Callback raised {type(e).__name__}: {str(e)[:300]}", []
        self.excluded.update(p1)
        self.excluded.update(p2)
        for k in sorted(set(c1) | set(c2)):
            if k not in c1 or k not in c2 or not identical(c1[k], c2[k]) or ((k in b1) != (k in b2)) or ((k in cat1) != (k in cat2)):
                self.excluded[k] = "Output not repeatable across two identical runs"
        stable = {k: v for k, v in c1.items() if k not in self.excluded}
        self.categorical_channels = sorted(k for k in cat1 if k in stable)
        why = "; ".join(f"{k}: {v}" for k, v in sorted(self.excluded.items()))[:900]
        return (stable if stable else None), why, sorted(k for k in b1 if k in stable)


@dataclasses.dataclass
class Result:
    status: str
    reason: Optional[str]
    settings: Settings
    input_rows: int
    channels: List[str]
    repeatability: str
    probes: List[dict]
    reducer: dict
    runs_used: int
    max_runs: int
    limitations: List[str]
    raw_prefix_recomputed: bool
    boolean_channels: List[str] = dataclasses.field(default_factory=list)
    categorical_channels: List[str] = dataclasses.field(default_factory=list)
    findings: List[dict] = dataclasses.field(default_factory=list)

    @property
    def divergences(self) -> List[dict]:
        return [p["first_divergence"] for p in self.probes if p["first_divergence"]]


def _probe_not_assessed(L: int, channels: Sequence[str], reason: str) -> dict:
    return {"prefix_length": L, "state": NOT_ASSESSED, "reason": reason[:1000], "compared_ranges": [],
            "skipped_ranges": [{"channel": c, "start": 0, "stop": L, "reason": R_PROBE_NA} for c in channels],
            "divergence_count": 0, "first_divergence": None}


def temporal_key(name) -> bool:
    """Explicit tokens/aliases, never substring matches such as sentiment/candidate."""
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(name))
    tokens = re.findall(r"[a-z]+", name.lower())
    return any(token in {"date", "time", "datetime", "timestamp", "epoch", "ts", "unix"}
               or re.fullmatch(r"unix(?:s|ms|us|ns|seconds|milliseconds|microseconds|nanoseconds)", token)
               for token in tokens)


def input_time_order(raw: pd.DataFrame) -> Tuple[Optional[str], List[str]]:
    """Validate index and every explicit temporal column without reordering/coercing user data.
    CSV date strings are inspected as ISO dates, not converted in the callback input.
    Ambiguous named date columns are refused rather than guessed as month/day-first."""
    limitations = []
    if not isinstance(raw.index, (pd.DatetimeIndex, pd.PeriodIndex)):
        limitations.append("Time order taken from row order; index chronology not verified because the index is not datetime-like. Temporal columns are checked separately when present; month-end probes require a datetime-like index.")
    if not raw.index.is_unique or not raw.index.is_monotonic_increasing or raw.index.to_frame(index=False).isna().any().any():
        return "Input index must be unique and increasing in chronological order; newest-first, shuffled, duplicate or missing index rows are refused. Sort raw data before computing features.", limitations
    for name in raw.columns:
        values = raw[name]
        if not isinstance(values, pd.Series):
            return f"Duplicate input column name {name!r}; time order cannot be verified.", limitations
        dtype = values.dtype
        temporal = pd.api.types.is_datetime64_any_dtype(dtype) or isinstance(dtype, pd.PeriodDtype)
        # A numeric temporal key needs only ordering, not a guessed epoch unit/timezone.
        named = temporal_key(name)
        nonnull = values.dropna()
        strings = pd.api.types.is_string_dtype(dtype) or dtype == object
        text = nonnull.astype(str)
        compact = bool(len(nonnull)) and text.str.fullmatch(r"[12][0-9]{7}").all()
        if not temporal and compact:
            try:
                values = pd.to_datetime(values.astype('string'), format="%Y%m%d", errors="raise", utc=True)
                temporal = True
            except (ValueError, TypeError, OverflowError):
                if named:
                    return f"Temporal column {name!r} is not valid YYYYMMDD data; parse it explicitly before checking.", limitations
        if named and not temporal and pd.api.types.is_numeric_dtype(dtype):
            if not pd.api.types.is_bool_dtype(dtype) and np.isfinite(values).all():
                temporal = True  # positive-unit epoch seconds/milliseconds/microseconds/nanoseconds
        if not temporal and strings:
            iso = bool(len(nonnull)) and all(isinstance(v, str) and len(v) >= 10 and v[:4].isdigit()
                and v[4] == '-' and v[5:7].isdigit() and v[7] == '-' and v[8:10].isdigit() for v in nonnull)
            if named and bool(len(nonnull)) and text.str.fullmatch(r"-?[0-9]+(?:\.[0-9]+)?").all():
                values = pd.to_numeric(values, errors="raise")
                temporal = bool(np.isfinite(values).all())
            elif named or iso:
                try:
                    values = pd.to_datetime(values, format="ISO8601", errors="raise", utc=True)
                    temporal = True
                except (ValueError, TypeError, OverflowError):
                    return f"Temporal column {name!r} is not unambiguous ISO datetime data; parse it explicitly before checking.", limitations
        if named and not temporal:
            return f"Temporal column {name!r} has unsupported dtype {dtype}; parse it explicitly before checking.", limitations
        if temporal and (values.isna().any() or not values.is_unique or not values.is_monotonic_increasing):
            return f"Temporal column {name!r} must be strictly increasing, unique and non-missing; newest-first, shuffled or duplicate times are refused before the callback.", limitations
    return None, limitations


def check(fn: FeatureFn, raw: pd.DataFrame, settings: Settings = Settings(), *, extra_limitations: Sequence[str] = (),
          raw_prefix_recomputed: bool = True, extra_not_assessed_reason: Optional[str] = None) -> Result:
    settings.validate()
    if not isinstance(raw, pd.DataFrame) or len(raw) < 2:
        raise ValueError("raw must be a DataFrame with at least 2 rows")
    n = len(raw)
    runner = Runner(fn, raw)
    cutoffs = choose_cutoffs(n, settings, raw)
    max_runs = 4 + 2 * len(cutoffs) + settings.reduce_budget
    limitations = list(extra_limitations)
    bool_channels: List[str] = []
    cat_channels: List[str] = []
    findings: List[dict] = []

    def done(status, reason, channels, repeat, probes, reducer=None):
        if runner.external.inputs:
            from .inputs import inventory_reason
            external_reason = inventory_reason(runner.external.inputs)
            channels = sorted(set(channels) | set(runner.external.inputs))
            for channel, detail in sorted(runner.external.inputs.items()):
                label = "Default-argument state" if channel.startswith("input:default:") else "Input"
                limitations.append(f"{label} NOT_ASSESSED {channel!r}: {detail}")
                for p in probes:
                    p["skipped_ranges"].append({"channel": channel, "start": 0, "stop": p["prefix_length"],
                        "reason": R_CHANNEL_NA + detail if p["state"] == "COMPARED" else R_PROBE_NA})
            for p in probes:
                if p["state"] == "SKIPPED_BOTH_NAN":
                    # There is now an unavailable input alongside the NaN-only outputs.
                    p.update(_probe_not_assessed(p["prefix_length"], channels, external_reason))
                elif p["state"] == "COMPARED":
                    p["reason"] = ((p["reason"] + "; " if p["reason"] else "") + external_reason)[:1000]
            if status == NO_DIVERGENCE:
                status, reason = NOT_ASSESSED, external_reason
            limitations.append(external_reason)
        return Result(status, reason, settings, n, channels, repeat, probes,
                      reducer or {"state": "NOT_ATTEMPTED", "method": None, "callback_runs": 0, "reproducer": None},
                      runner.runs, max_runs, limitations + sorted(runner.limitations), raw_prefix_recomputed and not bool(runner.external.inputs), list(bool_channels), list(cat_channels), findings)

    time_problem, time_limitations = input_time_order(raw)
    limitations.extend(time_limitations)
    if time_problem:
        return done(NOT_ASSESSED, time_problem, [], "NOT_ASSESSED",
                    [_probe_not_assessed(L, [], "Input time order refused.") for L in cutoffs])
    if settings.warmup_rows:
        limitations.append(f"Declared warm-up {settings.warmup_rows} is a hint only: all pre-terminal rows are audited, including warm-up values and missingness.")
    candidate_count = len(set(L for vals in cutoff_candidates(n, settings, raw).values() for L in vals))
    limitations.append(f"Prefix plan: {len(cutoffs)} selected of {candidate_count} grid/data-aware candidates (cap {MAX_PROBES}); unsampled rows can hide rare leaks.")
    full, why, bool_channels = runner.run_twice(0, n)
    cat_channels = list(runner.categorical_channels)
    baseline_exclusions = dict(runner.excluded)
    if full is None:
        repeat = "FAILED" if "not repeatable" in why else "NOT_ASSESSED"
        return done(NOT_ASSESSED, f"Full-history run: {why}", [], repeat, [_probe_not_assessed(L, [], "Baseline not assessed.") for L in cutoffs])
    channels = sorted(set(full) | set(baseline_exclusions))
    for c, reason in sorted(baseline_exclusions.items()):
        limitations.append(f"Channel NOT_ASSESSED {c!r}: {reason}")
    probes = []
    assessed_by_channel = {c: 0 for c in channels}
    partial = bool(baseline_exclusions)
    for L in cutoffs:
        pref, why, pbools = runner.run_twice(0, L)
        if pref is None:
            probes.append(_probe_not_assessed(L, channels, why)); continue
        cmp_r, skp_r, diffs, assessed, problems = [], [], [], 0, []
        for c in channels:
            excluded = baseline_exclusions.get(c) or runner.excluded.get(c)
            if c not in pref and not excluded:
                excluded = "Channel missing from prefix output"
            if c in full and c in pref and ((c in bool_channels) != (c in pbools)):
                excluded = "Boolean encoding differs from the full run"
            if c in full and c in pref and ((c in cat_channels) != (c in runner.categorical_channels)):
                excluded = "Categorical encoding differs from the full run"
            if excluded:
                skp_r.append({"channel": c, "start": 0, "stop": L, "reason": R_CHANNEL_NA + excluded[:800]})
                problems.append(f"{c}: {excluded}")
                continue
            r = compare_channel(c, full[c], pref[c], L, settings, categorical=c in cat_channels)
            if r["diffs"]:
                from .presentation import finding
                findings.append(finding(r["diffs"][0], L, c in cat_channels, fn, raw))
            cmp_r += r["compared"]; skp_r += r["skipped"]; diffs += r["diffs"]; assessed += r["assessed"]
            if r["unassessable"]:
                problems.append(r["unassessable"])
            assessed_by_channel[c] += r["assessed"]
        extra = sorted((set(pref) | set(runner.excluded)) - set(channels))
        if extra:
            problems.append(f"Extra prefix channels not in baseline: {extra[:10]}")
        partial = partial or bool(problems)
        if not assessed:
            if not problems and skp_r and all(r["reason"] in (R_INFERRED, R_MATCHED_NAN) for r in skp_r):
                probes.append({"prefix_length": L, "state": "SKIPPED_BOTH_NAN", "reason": "Only matching NaNs: warm-up or sparse signal, no numeric coverage.",
                               "compared_ranges": [], "skipped_ranges": skp_r, "divergence_count": 0, "first_divergence": None})
            else:
                probes.append(_probe_not_assessed(L, channels, "; ".join(problems) or "No assessable cell at this prefix."))
            continue
        first = min(diffs, key=lambda d: (d["row"], d["channel"])) if diffs else None
        probes.append({"prefix_length": L, "state": "COMPARED", "reason": "; ".join(problems)[:1000] or None,
                       "compared_ranges": cmp_r, "skipped_ranges": skp_r, "divergence_count": len(diffs), "first_divergence": first})

    diverged = [p for p in probes if p["divergence_count"]]
    unassessed = [p for p in probes if p["state"] == NOT_ASSESSED]
    nan_only = [p for p in probes if p["state"] == "SKIPPED_BOTH_NAN"]
    uncovered = [c for c, k in assessed_by_channel.items() if k == 0]
    if uncovered:
        limitations.append(f"Channels with no numeric coverage at any probe: {uncovered[:20]}")
    if nan_only:
        limitations.append(f"{len(nan_only)} prefixes skipped as matching NaN only (warm-up/sparse signal); not counted as agreement.")
    if unassessed:
        limitations.append(f"{len(unassessed)} of {len(probes)} planned probes not assessed; see their reasons.")
    if any(p["skipped_ranges"] for p in probes):
        limitations.append("Matching NaN cells and unavailable channels are skipped explicitly in replay.json; only compared_ranges count as numeric coverage.")
    reducer = None
    if diverged:
        status, reason = DIVERGENCE, None
        first = min((p["first_divergence"] for p in diverged), key=lambda d: (d["row"], d["channel"]))
        L0 = min(p["prefix_length"] for p in diverged if p["first_divergence"]["channel"] == first["channel"]
                 and p["first_divergence"]["row"] == first["row"])
        reducer = reduce_counterexample(runner, first["channel"], L0, n, settings)
    elif unassessed or partial or not any(assessed_by_channel.values()):
        status, reason = NOT_ASSESSED, f"{len(unassessed)} planned probe(s) not assessed; partial channels or absent numeric coverage: {partial or not any(assessed_by_channel.values())}."
    elif not raw_prefix_recomputed:
        status, reason = NOT_ASSESSED, extra_not_assessed_reason or "Raw-prefix recomputation not established."
    else:
        status, reason = NO_DIVERGENCE, None
    meaningful, trivial = meaningful_coverage(full, raw, probes, settings)
    for channel in cat_channels:
        if channel in meaningful:
            meaningful.remove(channel)
        trivial[channel] = "categorical equality coverage does not establish meaningful numeric per-bar coverage"
    if meaningful and status == NO_DIVERGENCE:
        # Two repeats on one deterministic counterfactual: reverse numeric data
        # values jointly, preserving the index, temporal keys, dtypes, OHLC
        # relationships, and missing-value locations. This is a dependency screen,
        # not another leak finding or a backtest. Never mutate the original rows.
        perturbed = raw.copy(deep=True)
        numeric = raw.select_dtypes(include=[np.number, "bool"]).columns
        mutable = []
        numeric_data = []
        for name in numeric:
            temporal = temporal_key(name)
            text = raw[name].dropna().astype(str)
            compact = len(text) and text.str.fullmatch(r"[12][0-9]{7}").all()
            if not temporal and not compact and not pd.api.types.is_complex_dtype(raw[name].dtype):
                unique = raw[name].nunique(dropna=True)
                if unique:
                    numeric_data.append(name)  # constant-valued columns can still have missingness
                if unique > 1:
                    mutable.append(name)
        permutation = np.arange(n)
        complete = np.array([], dtype=int)
        if mutable:
            # A shared permutation among complete rows preserves inter-column
            # constraints and missingness without coercing extension dtypes.
            complete = np.flatnonzero(raw[numeric_data].notna().all(axis=1).to_numpy())
            permutation[complete] = complete[::-1]
            for name in mutable:
                col = raw[name].copy(deep=True)
                col.iloc[complete] = raw[name].iloc[complete[::-1]].array
                perturbed[name] = col
        dep_runner = Runner(getattr(fn, "_nullcheck_fresh", lambda: fn)(), perturbed)
        changed, dep_why, dep_bools = dep_runner.run_twice(0, n)
        runner.runs += dep_runner.runs
        runner.external.inputs.update(dep_runner.external.inputs)
        # Exclusions from an artificial input do not become raw-input channels.
        # Instead they explain why this channel cannot establish dependency.
        survivors = []
        for channel in meaningful:
            if (changed is None or channel not in changed or ((channel in bool_channels) != (channel in dep_bools))
                    or ((channel in cat_channels) != (channel in dep_runner.categorical_channels))):
                trivial[channel] = "numeric-input dependency probe unavailable: " + (dep_why or "channel missing or encoding changed")[:500]
                continue
            compared = np.zeros(n, dtype=bool)
            for probe in probes:
                for region in probe["compared_ranges"]:
                    if region["channel"] == channel:
                        compared[region["start"]:region["stop"]] = True
            a, b = full[channel], changed[channel]
            finite = compared & np.isfinite(a) & np.isfinite(b)
            with np.errstate(over="ignore", invalid="ignore"):
                responds = finite & (np.abs(a - b) > settings.atol + settings.rtol * np.maximum(np.abs(a), np.abs(b)))
            # Sparse signals encode inactivity as NaN; a finite event disappearing
            # under changed data also establishes dependency, never leak evidence.
            responds |= compared & np.isfinite(a) & np.isnan(b)
            if not responds.any():
                trivial[channel] = "no observed dependency on numeric raw-input values in the bounded dependency probe"
                continue
            # Reuse f(Px), already computed above. If it agrees with P f(x),
            # row-local transforms cannot establish cross-row coverage. Require
            # positive non-equivariance evidence at actually compared output
            # positions. The permuted full output can include warm-up NaNs;
            # requiring source coverage too would reject long rolling windows.
            # Invalid infinities and matching NaNs cannot supply evidence.
            eligible = np.zeros(n, dtype=bool)
            eligible[complete] = True
            eligible &= compared
            permuted = a[permutation]
            fa, fb = np.isfinite(permuted), np.isfinite(b)
            with np.errstate(over="ignore", invalid="ignore"):
                mismatch = fa & fb & (np.abs(permuted - b) > settings.atol + settings.rtol * np.maximum(np.abs(permuted), np.abs(b)))
            mismatch |= (fa & np.isnan(b)) | (np.isnan(permuted) & fb)
            if (eligible & mismatch).any():
                survivors.append(channel)
            else:
                trivial[channel] = "row-local/permutation-equivariant on compared cells, or no observed cross-row dependency in the bounded reversal probe"
        meaningful = survivors
        limitations.append("Minimum coverage dependency probe: two repeat runs with complete numeric input rows reversed while index/time keys and missingness stay fixed. The same runs exclude outputs commuting with the row permutation; no response, no cross-row evidence or a callback failure is inconclusive, never a leak finding. Nonlinear counter/time-only signals may remain unassessed.")
    if meaningful and status == NO_DIVERGENCE and not runner.external.inputs:
        limitations.append(f"Minimum meaningful coverage: compared non-trivial channel(s) {meaningful!r}. Agreement qualifies only with observed numeric-input and cross-row dependency; this bounded check does not prove arbitrary provenance or use in trading decisions.")
    for channel, detail in trivial.items():
        limitations.append(f"Minimum meaningful coverage excludes {channel!r}: {detail}.")
    if not meaningful and status != DIVERGENCE:
        if reason:
            limitations.append("Additional assessment limitation: " + reason)
        status, reason = NOT_ASSESSED, R_MEANINGFUL
        limitations.insert(0, R_MEANINGFUL + ": no compared channel passed the derived, non-constant, non-pass-through, non-counter, non-row-local screen; compared channels remain listed.")
        # Schema 0.3 already represents a compared probe with incomplete coverage
        # using a reason. Preserve its numeric ranges; do not fabricate a skipped
        # channel or claim that raw-prefix recomputation failed.
        for probe in probes:
            if probe["state"] == "COMPARED":
                probe["reason"] = (R_MEANINGFUL + ("; " + probe["reason"] if probe["reason"] else ""))[:1000]
    if status == DIVERGENCE and not raw_prefix_recomputed:
        limitations.append(extra_not_assessed_reason or "Raw-prefix recomputation not established for all inputs.")
    return done(status, reason, channels, "PARTIAL" if baseline_exclusions else "CONFIRMED", probes, reducer)


# ---- reducer -----------------------------------------------------------------------------------------------------
def diverges(runner: Runner, channel: str, head: int, short: int, long: int, s: Settings) -> Optional[dict]:
    """Same scoped predicate on input rows [head, long): channel computed on the first (short-head) rows differs from
    the same channel computed on all rows, at an eligible row (warm-up/terminal rules applied to the reduced input),
    with both runs repeatable. Returns the first difference or None."""
    if not (head < short < long):
        return None
    original_fn = runner.fn
    fresh = getattr(original_fn, "_nullcheck_fresh", None)
    if fresh is not None:
        runner.fn = fresh()
    try:
        a, _, ba = runner.run_twice(head, long)
        ca = set(runner.categorical_channels)
        b, _, bb = runner.run_twice(head, short)
        cb = set(runner.categorical_channels)
    finally:
        runner.fn = original_fn
    if a is None or b is None or channel not in a or channel not in b or ((channel in ba) != (channel in bb)) or ((channel in ca) != (channel in cb)):
        return None
    r = compare_channel(channel, a[channel], b[channel], short - head, s, categorical=channel in ca)
    return min(r["diffs"], key=lambda d: d["row"]) if r["diffs"] else None


def reduce_counterexample(runner: Runner, channel: str, L: int, n: int, s: Settings) -> dict:
    """Bounded search for a smaller input [head, long) that re-triggers the same scoped predicate. Each predicate
    costs 4 callback runs; the final verification is inside the budget. Small, not minimal."""
    start = runner.runs
    cost = 4

    def left() -> bool:
        return runner.runs - start + cost <= s.reduce_budget

    def ok(h, sh, lg) -> bool:
        return left() and diverges(runner, channel, h, sh, lg, s) is not None

    head, short, long = 0, L, n
    lo, hi = 1, short          # 1. smaller truncated run
    while lo < hi and left():
        mid = (lo + hi) // 2
        if ok(head, mid, long):
            hi = mid
        else:
            lo = mid + 1
    if hi < short and ok(head, hi, long):
        short = hi
    lo, hi = short + 1, long                    # 2. fewer future rows
    while lo < hi and left():
        mid = (lo + hi) // 2
        if ok(head, short, mid):
            hi = mid
        else:
            lo = mid + 1
    if hi < long and ok(head, short, hi):
        long = hi
    lo, hi = 0, short - 1       # 3. less leading history
    while lo < hi and left():
        mid = (lo + hi + 1) // 2
        if ok(mid, short, long):
            lo = mid
        else:
            hi = mid - 1
    if lo and ok(lo, short, long):
        head = lo
    if not left():
        return {"state": "ATTEMPTED_NO_REDUCTION", "method": "bounded bisection (prefix, future rows, head)",
                "callback_runs": runner.runs - start, "reproducer": None,
                "note": "Budget exhausted before final verification; no reproducer is claimed."}
    d = diverges(runner, channel, head, short, long, s)
    reduced = (head, short, long) != (0, L, n)
    if d is None or not reduced:
        return {"state": "ATTEMPTED_NO_REDUCTION", "method": "bounded bisection (prefix, future rows, head)",
                "callback_runs": runner.runs - start, "reproducer": None,
                "note": "No smaller input re-triggered the predicate within the budget." if d is None or not reduced else ""}
    return {"state": "REPRODUCED_SMALLER_CASE", "method": "bounded bisection (prefix, future rows, head); verified",
            "callback_runs": runner.runs - start,
            "reproducer": {"channel": channel, "head": head, "prefix_stop": short, "stop": long,
                           "rows": long - head, "first_difference": d,
                           "note": "Small reproducer found, not guaranteed minimal. Rows are positions in reproducer.csv."}}

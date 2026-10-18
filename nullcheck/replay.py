"""Private replay record (schema 0.4.0; legacy 0.2.0/0.3.0 validation retained): write, validate, replay.

All file paths in a record are relative to an artifact ROOT chosen by the caller (default: the current directory).
Files outside the root get path=null and make reproduction UNAVAILABLE; nothing is copied to evade that. The record
folder (inside the root) holds: replay.json, input_snapshot.csv (CSV snapshot of the DataFrame that was checked),
config.json, environment.json, reproducer.csv/.json (if a smaller case was found), report.txt, PRIVATE_README.txt.
Your source files are referenced in place by hash. The folder is owner-only (0700/0600), never overwritten, never
uploaded.
"""
from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import inspect
import io
import json
import os
import platform
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import core

SCHEMA_VERSION = "0.4.0"
CHECK_ID = "indicator_prefix_invariance"
USER_PREFIX = "_nullcheck_user_"
SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "schema_0_4_0.json")


def _schema_path(version):
    filename = {"0.2.0": "schema_0_2_0.json", "0.3.0": "schema_0_3_0.json"}.get(version, "schema_0_4_0.json")
    return os.path.join(os.path.dirname(__file__), filename)

AVAILABILITY = ("Row i of a channel may use raw input rows 0..i only (the check's invariant). This is the declared "
                "assumption under test; the record does not prove when data became available or executable.")
CALENDAR = "Rows are taken as given (order, calendar, adjustments, timezone); nothing is resampled, adjusted or filled."


class EmissionError(RuntimeError):
    """A record failed validation at write time; no replay.json was written."""


# ---- data and code identity --------------------------------------------------------------------------------------
def data_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, float_format="%.17g", date_format="%Y-%m-%dT%H:%M:%S.%f%z")
    return buf.getvalue().encode("utf-8")


def frame_meta(df: pd.DataFrame) -> dict:
    return {"rows": len(df), "columns": [str(c) for c in df.columns],
            "dtypes": {str(c): str(t) for c, t in df.dtypes.items()},
            "index": {"name": df.index.name, "is_datetime": isinstance(df.index, pd.DatetimeIndex),
                      "tz": None if getattr(df.index, "tz", None) is None else str(df.index.tz)}}


def load_csv(path: str, meta: dict) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, float_precision="round_trip")
    if meta["index"]["is_datetime"]:
        df.index = pd.to_datetime(df.index, format="ISO8601")
        if meta["index"]["tz"] is None and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
    df.index.name = meta["index"]["name"]
    for c, t in meta["dtypes"].items():
        if c not in df.columns:
            continue
        if t.startswith(("float", "int", "bool")):
            df[c] = df[c].astype(t)
        elif t.startswith("datetime64["):
            df[c] = pd.to_datetime(df[c], format="ISO8601", utc="," in t).astype(t)
        elif t.startswith("period["):
            df[c] = pd.PeriodIndex(df[c], freq=t[7:-1])
    return df


def _round_trips(path: str, df: pd.DataFrame) -> bool:
    try:
        back = load_csv(path, frame_meta(df))
    except Exception:
        return False
    return (back.shape == df.shape and list(back.columns) == [str(c) for c in df.columns]
            and back.index.equals(df.index) and list(back.dtypes) == list(df.dtypes) and back.equals(df.set_axis([str(c) for c in df.columns], axis=1)))


MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_JSON_DEPTH = 16


def strict_load(path: str) -> Any:
    """Read record JSON under REPLAY_FORMAT parsing rules: UTF-8, <= 16 MiB, nesting depth <= 16, and reject
    duplicate keys, NaN/Infinity literals and numbers that decode to a non-finite float such as 1e999 (ValueError),
    before any semantic use."""
    if os.path.getsize(path) > MAX_JSON_BYTES:
        raise ValueError(f"{os.path.basename(path)}: larger than 16 MiB")

    def no_dupes(pairs):
        keys = [k for k, _ in pairs]
        if len(keys) != len(set(keys)):
            raise ValueError(f"{os.path.basename(path)}: duplicate key(s) {sorted({k for k in keys if keys.count(k) > 1})[:5]}")
        return dict(pairs)

    def no_constant(name):
        raise ValueError(f"{os.path.basename(path)}: non-finite literal {name} is not allowed")

    def depth(x, d=1):
        if d > MAX_JSON_DEPTH:
            raise ValueError(f"{os.path.basename(path)}: nesting deeper than {MAX_JSON_DEPTH}")
        for v in (x.values() if isinstance(x, dict) else x if isinstance(x, list) else ()):
            depth(v, d + 1)

    def finite_float(text):
        v = float(text)
        if not np.isfinite(v):
            raise ValueError(f"{os.path.basename(path)}: number {text[:40]} is not finite once decoded")
        return v

    with open(path, encoding="utf-8") as f:
        obj = json.load(f, object_pairs_hook=no_dupes, parse_constant=no_constant, parse_float=finite_float)
    depth(obj)
    return obj


def sha256_file(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def load_file_module(path: str, fresh: bool = False) -> Any:
    """Import a .py file under a name unique to its absolute path. fresh=True re-executes it (used by replay after the
    file's hash was verified, so a stale in-process copy is never used)."""
    path = os.path.realpath(path)
    name = USER_PREFIX + hashlib.sha256(path.encode()).hexdigest()[:12]
    if fresh:
        sys.modules.pop(name, None)
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        del sys.modules[name]
        raise
    return mod


def load_by_spec(spec: str) -> Any:
    """'path/to/file.py:Name' or 'package.module:Name' -> object."""
    target, _, name = spec.partition(":")
    if not name:
        raise ValueError(f"expected FILE.py:NAME or MODULE:NAME, got {spec!r}")
    obj = load_file_module(target) if target.endswith(".py") else importlib.import_module(target)
    for part in name.split("."):
        obj = getattr(obj, part)
    return obj


def source_of(obj: Any) -> Tuple[Optional[str], Optional[str]]:
    """(defining file, reason-if-unavailable)."""
    if obj is None:
        return None, None
    if getattr(obj, "__module__", None) == "__main__":
        return None, "defined in a __main__ script or notebook; move it into an importable .py file to make replay available"
    if "<" in getattr(obj, "__qualname__", "<"):
        return None, "a lambda or nested function cannot be re-imported; define it at module level"
    try:
        path = inspect.getsourcefile(obj)
    except TypeError:
        path = None
    if not path or not os.path.isfile(path):
        return None, "source file not found"
    return os.path.abspath(path), None


# ---- private folder ----------------------------------------------------------------------------------------------
def _write(root: str, rel: str, data: bytes) -> str:
    path = os.path.join(root, rel)
    os.makedirs(os.path.dirname(path), mode=0o700, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    return hashlib.sha256(data).hexdigest()


def environment(framework: Optional[Tuple[str, str]]) -> List[str]:
    env = [f"python {sys.version.split()[0]}", f"numpy {np.__version__}", f"pandas {pd.__version__}",
           f"platform {platform.platform()}"]
    if framework:
        env.append(f"{framework[0]} {framework[1]}")
    env.append("Not recorded: other packages your code imports, and helper modules outside the copied source file(s).")
    return env


def _rel(path: str, root: str) -> Optional[str]:
    """Path relative to root if the resolved path is inside it, else None."""
    real = os.path.realpath(path)
    if real == root or not real.startswith(root + os.sep):
        return None
    rel = os.path.relpath(real, root)
    return rel if "\\" not in rel and ":" not in rel else None


def write_record(out_dir: str, res: core.Result, raw: pd.DataFrame, target: Dict[str, Any], adapter: dict,
                 framework: Optional[Tuple[str, str]], report_text: str, original_input: Optional[str] = None,
                 root: Optional[str] = None) -> dict:
    """Write the private folder; returns the record. target: {'kind', 'objects': {role: obj}, 'params'?, 'backtest_kwargs'?}."""
    from . import __version__
    root = os.path.realpath(root or os.getcwd())
    os.makedirs(out_dir, mode=0o700, exist_ok=False)
    os.chmod(out_dir, 0o700)
    files: List[dict] = []
    unavailable: List[str] = []
    if any(c.startswith("input:") and not c.startswith("input:default:") for c in res.channels):
        unavailable.append("external precomputed inputs were observed but not copied or hashed; their independent provenance is not replayable")
    if any(c.startswith("input:default:") for c in res.channels):
        unavailable.append("default-argument state was observed but its runtime contents were not captured; exact replay is unavailable")
    out_rel = _rel(out_dir, root)
    if out_rel is None:
        unavailable.append("the record folder is outside the chosen root")

    def put(fid: str, role: str, name: str, data: bytes) -> None:
        files.append({"id": fid, "role": role, "sha256": _write(out_dir, name, data),
                      "path": f"{out_rel}/{name}" if out_rel else None})

    snap = data_to_csv_bytes(raw)
    put("input_snapshot", "INPUT", "input_snapshot.csv", snap)
    if not _round_trips(os.path.join(out_dir, "input_snapshot.csv"), raw):
        unavailable.append("the input does not round-trip exactly through the CSV snapshot (values, dtypes or index "
                           "change when re-read, e.g. strings like 'NA' or mixed types); replay would use different data")
    if original_input:
        rel = _rel(original_input, root)
        files.append({"id": "original_input", "role": "INPUT", "sha256": sha256_file(original_input), "path": rel})

    cfg_target: Dict[str, Any] = {"kind": target["kind"]}
    seen: Dict[str, str] = {}
    for role, obj in target["objects"].items():
        if obj is None:
            cfg_target[role] = None
            continue
        path, why = source_of(obj)
        rel = _rel(path, root) if path else None
        if path and rel is None:
            why = "source file is outside the chosen root"
        if path and path not in seen:
            fid = f"source_{len(seen)}"
            files.append({"id": fid, "role": "SOURCE", "sha256": sha256_file(path), "path": rel})
            seen[path] = fid
        if why:
            unavailable.append(f"{role}: {why}")
        cfg_target[role] = {"qualname": getattr(obj, "__qualname__", repr(obj)), "source_file_id": seen.get(path)}
    for k in ("params", "backtest_kwargs"):
        if k in target:
            cfg_target[k] = target[k]
    config = {"target": cfg_target, "settings": dataclasses.asdict(res.settings), "input_meta": frame_meta(raw)}
    try:
        cfg_bytes = (json.dumps(config, indent=1, allow_nan=False) + "\n").encode()
    except (TypeError, ValueError) as e:
        unavailable.append(f"parameters are not plain JSON values ({str(e)[:200]})")
        cfg_bytes = (json.dumps(config, indent=1, allow_nan=False, default=str) + "\n").encode()
    put("config", "CONFIG", "config.json", cfg_bytes)
    env = environment(framework)
    put("environment", "ENVIRONMENT", "environment.json", (json.dumps(env, indent=1) + "\n").encode())

    red = res.reducer
    rep_id = None
    if red.get("reproducer"):
        rp = red["reproducer"]
        sl = raw.iloc[rp["head"]:rp["stop"]]
        put("reproducer_rows", "REPRODUCER", "reproducer.csv", data_to_csv_bytes(sl))
        desc = {"rows_file_id": "reproducer_rows", "input_meta": frame_meta(sl), "channel": rp["channel"],
                "prefix_rows": rp["prefix_stop"] - rp["head"], "rows": rp["rows"],
                "original_rows": [rp["head"], rp["stop"]], "first_difference": rp["first_difference"],
                "predicate": "channel computed on the first prefix_rows rows of reproducer.csv differs from the same "
                             "channel computed on all its rows, at an eligible row, both runs repeatable, same "
                             "warm-up/terminal/tolerance settings as config.json",
                "note": rp["note"]}
        put("reproducer", "REPRODUCER", "reproducer.json", (json.dumps(desc, indent=1, allow_nan=False) + "\n").encode())
        rep_id = "reproducer"

    rec = {
        "schema_version": SCHEMA_VERSION,
        "check": {"id": CHECK_ID, "version": __version__, "runner_version": __version__},
        "adapter": adapter,
        "status": res.status,
        "reason": res.reason,
        "contract": {
            "comparison_scope": "prefix_valid",
            "raw_prefix_recomputed": res.raw_prefix_recomputed,
            "reset_description": ("Each run gets a fresh deep copy of the raw rows. Global, class, cache and random-generator "
                                  "state of your code is NOT reset; two repeat runs per input detect instability, they do not prevent it."),
            "seed": res.settings.seed,
            "repeatability": res.repeatability,
            "warmup_rows": res.settings.warmup_rows,
            "terminal_excluded_rows": res.settings.terminal_excluded_rows,
            "atol": res.settings.atol, "rtol": res.settings.rtol,
            "availability_assumption": AVAILABILITY,
            "calendar_adjustment_assumption": CALENDAR,
            "limitations": [x[:1000] for x in (res.limitations + [
                "Seed selects the sampled prefix lengths only; it does not seed your code.",
                "Features precomputed before the callback, cached on disk or read from outside the given rows are not covered.",
            ])][:100],
            "warmup_policy": "AUDIT_ALL_ROWS_SKIP_MATCHING_NAN",
            "boolean_channels": res.boolean_channels,
            "categorical_channels": res.categorical_channels,
            "categorical_encoding": "BLAKE2B_53_CANONICAL_V1_EXACT_COLLISION_CHECKED",
        },
        "coverage": {"input_rows": res.input_rows, "channels": res.channels or ["(none)"],
                     "planned_prefix_lengths": [p["prefix_length"] for p in res.probes] or [1],
                     "max_callback_runs": res.max_runs, "actual_callback_runs": res.runs_used},
        "probes": res.probes,
        "files": files,
        "reproduction": {
            "availability": "UNAVAILABLE" if unavailable else "AVAILABLE",
            "reason": "; ".join(unavailable)[:1000] if unavailable else None,
            "argv": [] if unavailable else ["python", "-m", "nullcheck", "replay", out_rel, "--root", "."],
            "environment": env,
            "reducer": {"state": red["state"], "method": red["method"], "callback_runs": red["callback_runs"],
                        "reproducer_file_id": rep_id},
        },
        "privacy": {"visibility": "PRIVATE", "automatic_upload": False},
    }
    problems = validate_record(rec, root if out_rel else None)
    if problems:  # never leave a usable-looking record behind
        _write(out_dir, "EMISSION_INVALID.json", (json.dumps({"problems": problems}, indent=1) + "\n").encode())
        raise EmissionError("the emitted record failed validation: " + "; ".join(problems[:5]))
    _write(out_dir, "replay.json", (json.dumps(rec, indent=1, allow_nan=False) + "\n").encode())
    _write(out_dir, "report.txt", report_text.encode())
    _write(out_dir, "findings.json", (json.dumps({
        "purpose": "Presentation supplement, not independently certified by record validation. One first differing row per channel and prefix; comparison decisions remain in replay.json.",
        "findings": res.findings,
    }, indent=1, allow_nan=False) + "\n").encode())
    _write(out_dir, "PRIVATE_README.txt", (
        "Written by nullcheck on your machine. input_snapshot.csv and reproducer.csv are YOUR data rows; replay.json and\n"
        "config.json name your code files (relative to the root you chose), functions, settings, row labels, and may quote\n"
        "error messages from your code. Nothing was uploaded. Do not share this folder, not even with the authors: for\n"
        "support, build a separate synthetic example from scratch. Replaying this folder imports and runs the referenced\n"
        "code with your permissions; only replay records and code you trust.\n").encode())
    return rec


# ---- validation --------------------------------------------------------------------------------------------------
HARD = ("schema", "semantic", "path", "consistency")  # never overridable; only "hash" problems are


def _legacy_probe_ranges_ok(p: dict, chans: List[str], ct: dict) -> List[str]:
    """Per-probe range rules: partition of [0, L) per channel; COMPARED probes may only skip declared warm-up,
    the leading inferred warm-up and the declared terminal exclusion, and must compare at least one range."""
    errs, L = [], p["prefix_length"]
    w = min(ct["warmup_rows"], L)
    hi = max(w, L - ct["terminal_excluded_rows"])
    for c in chans:
        cmp_ = sorted((r["start"], r["stop"]) for r in p["compared_ranges"] if r["channel"] == c)
        skp = sorted((r["start"], r["stop"], r["reason"]) for r in p["skipped_ranges"] if r["channel"] == c)
        spans = sorted([(a, b) for a, b in cmp_] + [(a, b) for a, b, _ in skp])
        pos = 0
        for st, sp in spans:
            if st != pos or not st < sp <= L:
                errs.append(f"semantic: probe {L} channel {c}: ranges do not partition [0,{L})"); break
            pos = sp
        else:
            if pos != L:
                errs.append(f"semantic: probe {L} channel {c}: ranges do not cover [0,{L})")
        if p["state"] != "COMPARED":
            if cmp_ or skp != [(0, L, core.R_PROBE_NA)]:
                errs.append(f"semantic: probe {L} channel {c}: unassessed probe must skip [0,{L}) as not assessed")
            continue
        if len(cmp_) > 1:
            errs.append(f"semantic: probe {L} channel {c}: more than one compared range")
        expect = []
        if w > 0:
            expect.append((0, w, core.R_DECLARED))
        inferred = [x for x in skp if x[2] == core.R_INFERRED]
        if len(inferred) > 1 or (inferred and inferred[0][0] != w):
            errs.append(f"semantic: probe {L} channel {c}: inferred warm-up must be one leading run starting at {w}")
        end_lead = inferred[0][1] if inferred else w
        expect += inferred[:1]
        if hi < L:
            expect.append((hi, L, core.R_TERMINAL))
        if sorted(expect) != skp:
            errs.append(f"semantic: probe {L} channel {c}: skipped ranges other than declared/inferred warm-up and terminal exclusion")
        if cmp_ and cmp_[0] != (end_lead, hi):
            errs.append(f"semantic: probe {L} channel {c}: compared range must be [{end_lead},{hi})")
        if not cmp_ and end_lead != hi:
            errs.append(f"semantic: probe {L} channel {c}: rows [{end_lead},{hi}) neither compared nor excluded")
    if p["state"] == "COMPARED" and not p["compared_ranges"]:
        errs.append(f"semantic: probe {L}: COMPARED probe without any compared range")
    return errs


def _probe_ranges_ok(p: dict, chans: List[str], ct: dict) -> List[str]:
    if ct["warmup_policy"] == "DECLARED_AND_LEADING_BOTH_NAN":
        return _legacy_probe_ranges_ok(p, chans, ct)
    errs, L = [], p["prefix_length"]
    hi = max(0, L - ct["terminal_excluded_rows"])
    for c in chans:
        cmp_ = [r for r in p["compared_ranges"] if r["channel"] == c]
        skp = [r for r in p["skipped_ranges"] if r["channel"] == c]
        pos = 0
        for r in sorted(cmp_ + skp, key=lambda r: r["start"]):
            if r["start"] != pos or not pos < r["stop"] <= L:
                errs.append(f"semantic: probe {L} channel {c}: ranges do not partition [0,{L})")
                break
            pos = r["stop"]
        if pos != L:
            errs.append(f"semantic: probe {L} channel {c}: ranges do not cover [0,{L})")
        if p["state"] in ("NOT_ASSESSED", "ERROR"):
            if cmp_ or skp != [{"channel": c, "start": 0, "stop": L, "reason": core.R_PROBE_NA}]:
                errs.append(f"semantic: probe {L}: unassessed probe must skip entire channel")
            continue
        for r in cmp_:
            if r["stop"] > hi:
                errs.append(f"semantic: probe {L}: comparison extends into terminal exclusion")
        for r in skp:
            why = r["reason"]
            if why.startswith(core.R_CHANNEL_NA):
                if r["start"] != 0 or r["stop"] != L or cmp_ or len(skp) != 1 or not p["reason"]:
                    errs.append(f"semantic: probe {L}: invalid channel exclusion")
            elif why == core.R_TERMINAL:
                if (r["start"], r["stop"]) != (hi, L):
                    errs.append(f"semantic: probe {L}: invalid terminal exclusion")
            elif why in (core.R_INFERRED, core.R_MATCHED_NAN, core.R_INVALID):
                if r["stop"] > hi or (why == core.R_INFERRED and r["start"] != 0):
                    errs.append(f"semantic: probe {L}: invalid missing/invalid range")
                if why == core.R_INVALID and not p["reason"]:
                    errs.append(f"semantic: probe {L}: invalid numeric exclusion needs a reason")
            else:
                errs.append(f"semantic: probe {L}: skipped ranges other than documented missing/invalid/channel/terminal exclusions")
        if hi < L and not any(r["reason"].startswith(core.R_CHANNEL_NA) for r in skp):
            if not any(r["reason"] == core.R_TERMINAL for r in skp):
                errs.append(f"semantic: probe {L}: terminal exclusion missing")
    if p["state"] == "COMPARED" and not p["compared_ranges"]:
        errs.append(f"semantic: probe {L}: COMPARED probe without any compared range")
    if p["state"] == "SKIPPED_BOTH_NAN":
        if p["compared_ranges"] or not chans or not p["skipped_ranges"] or any(r["reason"] not in (core.R_INFERRED, core.R_MATCHED_NAN) for r in p["skipped_ranges"]):
            errs.append(f"semantic: probe {L}: SKIPPED_BOTH_NAN must contain only matching NaN ranges")
    return errs


def _legacy_cutoffs(n, s, raw):
    lo, hi = s.warmup_rows + 1, n - 1
    if hi < lo:
        return []
    base = np.linspace(lo, hi, num=max(1, s.n_cutoffs - s.n_cutoffs // 2)).round().astype(int).tolist()
    base += np.random.default_rng(s.seed).integers(lo, hi + 1, size=s.n_cutoffs // 2).tolist()
    chosen = sorted(set(int(x) for x in base))[:128]
    if s.nan_cutoffs:
        na = raw.isna().any(axis=1).to_numpy()
        for st in np.flatnonzero(na & ~np.r_[False, na[:-1]]):
            L = int(st) + 1
            if lo <= L <= hi and L not in chosen and len(chosen) < 128:
                chosen.append(L)
    return sorted(chosen)


def _check_reproducer(rec: dict, rid: str, paths: Dict[str, str]) -> List[str]:
    """Strictly parse and check reproducer.json and its rows file (called before any code import)."""
    if rid not in paths:
        return ["consistency: reproducer descriptor has no readable path"]
    try:
        desc = strict_load(paths[rid])
    except (ValueError, OSError) as e:
        return [f"consistency: reproducer.json rejected ({str(e)[:200]})"]
    need = {"rows_file_id": str, "input_meta": dict, "channel": str, "prefix_rows": int, "rows": int,
            "original_rows": list, "first_difference": dict}
    errs = [f"consistency: reproducer.json field {k!r} missing or not {t.__name__}" for k, t in need.items()
            if not isinstance(desc.get(k), t) or isinstance(desc.get(k), bool)]
    if errs:
        return errs
    role = {f["id"]: f["role"] for f in rec["files"]}
    if role.get(desc["rows_file_id"]) != "REPRODUCER" or desc["rows_file_id"] not in paths:
        errs.append("consistency: reproducer rows file is not a recorded REPRODUCER file")
    if desc["channel"] not in rec["coverage"]["channels"]:
        errs.append("consistency: reproducer channel is not a recorded channel")
    if not 1 <= desc["prefix_rows"] < desc["rows"]:
        errs.append("consistency: reproducer prefix_rows must be in [1, rows)")
    try:
        import jsonschema
        schema_path = _schema_path(rec.get("schema_version"))
        with open(schema_path) as f:
            diff_schema = dict(json.load(f)["$defs"]["difference"])
        errs += [f"consistency: reproducer first_difference: {e.message}"
                 for e in jsonschema.Draft202012Validator(diff_schema).iter_errors(desc["first_difference"])]
    except ImportError:
        errs.append("schema: jsonschema is required for record validation but is not installed")
    return errs


def _consistency(rec: dict, root: str, paths: Dict[str, str]) -> List[str]:
    """Cross-file checks: config.json and the input snapshot must reproduce the record's contract and plan."""
    errs: List[str] = []
    ct, cov = rec["contract"], rec["coverage"]
    if "config" not in paths:
        return []
    try:
        cfg = strict_load(paths["config"])
        st = core.Settings(**cfg["settings"])
        st.validate()
    except (ValueError, TypeError, KeyError, OSError) as e:
        return [f"consistency: config.json unusable ({str(e)[:200]})"]
    for k in ("warmup_rows", "terminal_excluded_rows", "atol", "rtol", "seed"):
        if getattr(st, k) != ct[k] or type(getattr(st, k)) is bool:
            errs.append(f"consistency: contract.{k}={ct[k]!r} but config.json has {getattr(st, k)!r}")
    kind = cfg.get("target", {}).get("kind")
    if {"callable": "python_callback", "backtesting.py": "backtesting.py"}.get(kind) != rec["adapter"]["id"]:
        errs.append(f"consistency: adapter {rec['adapter']['id']!r} does not match config target kind {kind!r}")
    src_ids = {f["id"] for f in rec["files"] if f["role"] == "SOURCE"}
    for role, ref in cfg.get("target", {}).items():
        if isinstance(ref, dict) and "source_file_id" in ref and ref["source_file_id"] and ref["source_file_id"] not in src_ids:
            errs.append(f"consistency: config {role} names an unknown SOURCE file")
    rid = rec["reproduction"]["reducer"]["reproducer_file_id"]
    if rid:
        errs += _check_reproducer(rec, rid, paths)
    if "input_snapshot" in paths:
        try:
            raw = load_csv(paths["input_snapshot"], cfg["input_meta"])
        except Exception as e:  # malformed snapshot
            return errs + [f"consistency: input snapshot unreadable ({type(e).__name__})"]
        if len(raw) != cov["input_rows"]:
            errs.append("consistency: coverage.input_rows differs from the input snapshot")
        planned = (_legacy_cutoffs if rec["schema_version"] == "0.2.0" else core.choose_cutoffs)(len(raw), st, raw)
        if rec["probes"] and planned != cov["planned_prefix_lengths"]:
            errs.append("consistency: planned_prefix_lengths are not those config.json produces on the input snapshot")
        dependency_budget = 0 if rec["check"]["runner_version"] in ("0.1.0", "0.1.1", "0.2.0", "0.2.1", "0.2.2") else 2
        if rec["probes"] and cov["max_callback_runs"] != 2 + dependency_budget + 2 * len(planned) + st.reduce_budget:
            errs.append("consistency: max_callback_runs does not follow from config.json")
    return errs


def validate_record(rec: dict, root: Optional[str] = None) -> List[str]:
    """Full validation of a 0.4.0, 0.3.0 or legacy 0.2.0 record. Every problem is prefixed with its class: schema / semantic / path /
    consistency (never overridable) or hash (a recorded file changed; overridable only by explicit modified replay).
    jsonschema is a required dependency: if it is missing, validation fails rather than silently weakening.
    With root, file paths are resolved against it (containment, existence, hashes) and cross-file consistency of
    config.json and the input snapshot with the record is checked."""
    try:
        import jsonschema
    except ImportError:
        return ["schema: jsonschema is required for record validation but is not installed"]
    schema_path = _schema_path(rec.get("schema_version"))
    with open(schema_path) as f:
        schema = json.load(f)
    errs = [f"schema: {e.message}" for e in jsonschema.Draft202012Validator(schema).iter_errors(rec)]
    if errs:
        return errs
    cov, probes = rec["coverage"], rec["probes"]
    chans = cov["channels"] if cov["channels"] != ["(none)"] else []
    Ls = [p["prefix_length"] for p in probes]
    if Ls != sorted(set(Ls)):
        errs.append("semantic: probes not sorted/unique by prefix_length")
    if probes and Ls != cov["planned_prefix_lengths"]:
        errs.append("semantic: one probe per planned prefix length required")
    if any(not 1 <= L < cov["input_rows"] for L in Ls):
        errs.append("semantic: prefix length outside 1..input_rows-1")
    if cov["actual_callback_runs"] > cov["max_callback_runs"]:
        errs.append("semantic: actual_callback_runs exceeds max_callback_runs")
    ct = rec["contract"]
    rt, at = ct["rtol"], ct["atol"]
    for p in probes:
        L = p["prefix_length"]
        if p["state"] != "COMPARED" and (p["divergence_count"] or p["first_divergence"] or not p["reason"]):
            errs.append(f"semantic: probe {L}: unassessed probe must have no findings and a reason")
        errs += _probe_ranges_ok(p, chans, ct)
        for r in p["compared_ranges"] + p["skipped_ranges"]:
            if r["channel"] not in chans:
                errs.append(f"semantic: probe {L}: range for unknown channel {r['channel']!r}")
        fd = p["first_divergence"]
        if p["state"] == "COMPARED":
            if (p["divergence_count"] > 0) != (fd is not None):
                errs.append(f"semantic: probe {L}: divergence_count and first_divergence disagree")
            if fd:
                inside = any(r["channel"] == fd["channel"] and r["start"] <= fd["row"] < r["stop"] for r in p["compared_ranges"])
                if not inside:
                    errs.append(f"semantic: probe {L}: first_divergence outside compared ranges")
                a, b = fd["full_value"], fd["prefix_value"]
                for v, stt in ((a, fd["full_state"]), (b, fd["prefix_state"])):
                    if (stt == "FINITE") != (v is not None):
                        errs.append(f"semantic: probe {L}: value/state pairing violated")
                if fd["kind"] == "VALUE":
                    categorical = fd["channel"] in ct.get("categorical_channels", [])
                    differs = (a != b) if categorical else (a is not None and b is not None and abs(a - b) > at + rt * max(abs(a), abs(b)))
                    if a is None or b is None or not differs:
                        errs.append(f"semantic: probe {L}: VALUE divergence does not breach {'categorical equality' if categorical else 'tolerance'}")
                elif (fd["full_state"] == "FINITE") == (fd["prefix_state"] == "FINITE"):
                    errs.append(f"semantic: probe {L}: MISSINGNESS divergence needs exactly one FINITE side")
    st = rec["status"]
    any_div = any(p["state"] == "COMPARED" and p["divergence_count"] for p in probes)
    any_na = any(p["state"] != "COMPARED" for p in probes)
    if rec["schema_version"] in ("0.3.0", "0.4.0"):
        any_na = any(p["state"] in ("NOT_ASSESSED", "ERROR") or (p["state"] == "COMPARED" and p["reason"]) for p in probes)
        any_na = any_na or not any(p["compared_ranges"] for p in probes)
    if any(p["state"] == "ERROR" for p in probes) and st != "ERROR":
        errs.append("semantic: status must be ERROR when a probe errored")
    elif st != "ERROR":
        expect = ("DIVERGENCE_OBSERVED" if any_div else
                  "NOT_ASSESSED" if (any_na or not probes or ct["repeatability"] != "CONFIRMED"
                                     or not ct["raw_prefix_recomputed"]) else "NO_DIVERGENCE_OBSERVED")
        if st != expect:
            errs.append(f"semantic: status {st} contradicts precedence (expected {expect})")
    if st in ("NOT_ASSESSED", "ERROR") and not rec["reason"]:
        errs.append("semantic: non-success status requires a reason")
    ids = [f["id"] for f in rec["files"]]
    if len(ids) != len(set(ids)):
        errs.append("semantic: duplicate file ids")
    rep = rec["reproduction"]
    if rep["availability"] == "AVAILABLE":
        roles = {f["role"] for f in rec["files"] if f["path"]}
        if not rep["argv"] or not {"INPUT", "SOURCE", "CONFIG"} <= roles:
            errs.append("semantic: AVAILABLE needs argv and INPUT, SOURCE and CONFIG files with paths")
    elif not rep["reason"]:
        errs.append("semantic: UNAVAILABLE needs a reason")
    if rep["reducer"]["reproducer_file_id"] and rep["reducer"]["reproducer_file_id"] not in ids:
        errs.append("semantic: reducer.reproducer_file_id is not a file id")
    if not set(ct["boolean_channels"]) <= set(chans):
        errs.append("semantic: boolean_channels must name known channels")
    cats = set(ct.get("categorical_channels", []))
    if not cats <= set(chans) or cats.intersection(ct["boolean_channels"]):
        errs.append("semantic: categorical_channels must name known, non-boolean channels")
    for probe in probes:
        fd = probe["first_divergence"]
        if fd and fd["channel"] in cats:
            for key in ("full_value", "prefix_value"):
                value = fd[key]
                if value is not None and (not 0 <= value < 2**53 or int(value) != value):
                    errs.append("semantic: categorical finite values must be exact 53-bit integer fingerprints")
    if root and not errs:
        root = os.path.realpath(root)
        paths: Dict[str, str] = {}
        for f in rec["files"]:
            if not f["path"]:
                continue
            p = os.path.realpath(os.path.join(root, f["path"]))
            if not p.startswith(root + os.sep):
                errs.append(f"path: file {f['id']} resolves outside the chosen root")
            elif not os.path.isfile(p):
                errs.append(f"path: file {f['id']} is missing")
            else:
                paths[f["id"]] = p
                if sha256_file(p) != f["sha256"]:
                    errs.append(f"hash: file {f['id']} differs from the recorded sha256")
        if not any(e.startswith("path:") for e in errs):
            errs += _consistency(rec, root, paths)
    return errs


# ---- replay ------------------------------------------------------------------------------------------------------
def replay(folder: str, fn: Optional[Callable] = None, allow_modified: bool = False, root: Optional[str] = None) -> dict:
    """Re-run the recorded check from verified files (paths resolved against root, default the current directory)
    and compare the whole recomputed result (status, reason, channels, encodings, every probe with its ranges) with
    the record. schema/semantic/path/consistency problems always refuse. Changed file hashes refuse unless
    allow_modified=True; passing fn also counts as modified. Modified runs are labelled MODIFIED_."""
    from .adapters import build_target_fn
    root = os.path.realpath(root or os.getcwd())
    rec = strict_load(os.path.join(folder, "replay.json"))
    if rec.get("schema_version") not in ("0.2.0", "0.3.0", SCHEMA_VERSION):
        return {"outcome": "UNSUPPORTED_SCHEMA", "schema_version": rec.get("schema_version")}
    problems = validate_record(rec, root)
    hard = [e for e in problems if e.split(":", 1)[0] in HARD]
    if hard or (problems and not allow_modified):
        return {"outcome": "REFUSED", "problems": problems}
    if rec["schema_version"] == "0.2.0":
        return {"outcome": "UNSUPPORTED_RUNNER", "reason": "Validated legacy record: exact replay requires its original nullcheck 0.1.x runner; v0.2 changes comparison and planning semantics."}
    if rec["schema_version"] == "0.3.0" or rec["check"]["runner_version"] in ("0.2.0", "0.2.1", "0.2.2", "0.2.3"):
        return {"outcome": "UNSUPPORTED_RUNNER", "reason": "Validated older record: exact replay requires its original nullcheck runner; v0.2.4 adds exact categorical equality comparison."}
    modified = fn is not None or bool(problems)
    if rec["reproduction"]["availability"] != "AVAILABLE" and fn is None:
        return {"outcome": "UNAVAILABLE", "reason": rec["reproduction"]["reason"]}
    paths = {f["id"]: os.path.realpath(os.path.join(root, f["path"])) for f in rec["files"] if f["path"]}
    cfg = strict_load(paths["config"])
    raw = load_csv(paths["input_snapshot"], cfg["input_meta"])
    if fn is None:
        fn = build_target_fn(cfg["target"], paths, root)
    settings = core.Settings(**cfg["settings"])
    rpr = rec["contract"]["raw_prefix_recomputed"]
    res = core.check(fn, raw, settings, raw_prefix_recomputed=rpr, extra_not_assessed_reason=None if rpr else rec["reason"])

    def view(status, reason, channels, bools, cats, repeat, probes):
        return json.loads(json.dumps({"status": status, "reason": reason, "channels": channels, "boolean_channels": bools, "categorical_channels": cats,
                                      "repeatability": repeat, "probes": probes}, allow_nan=False))
    got = view(res.status, res.reason, res.channels or ["(none)"], res.boolean_channels, res.categorical_channels, res.repeatability, res.probes)
    want = view(rec["status"], rec["reason"], rec["coverage"]["channels"], rec["contract"]["boolean_channels"], rec["contract"]["categorical_channels"],
                rec["contract"]["repeatability"], rec["probes"])
    diff = sorted(k for k in want if got[k] != want[k])
    out = {"outcome": ("MODIFIED_" if modified else "") + ("REPRODUCED" if not diff else "DIFFERENT"),
           "recorded_status": rec["status"], "replayed_status": res.status, "differing_fields": diff,
           "problems": problems}
    rid = rec["reproduction"]["reducer"]["reproducer_file_id"]
    if rid:
        desc = strict_load(paths[rid])
        rows = load_csv(paths[desc["rows_file_id"]], desc["input_meta"])
        d = core.diverges(core.Runner(fn, rows), desc["channel"], 0, desc["prefix_rows"], len(rows), settings)
        out["small_reproducer"] = "REPRODUCED" if d == desc["first_difference"] else "DIFFERENT"
    return out

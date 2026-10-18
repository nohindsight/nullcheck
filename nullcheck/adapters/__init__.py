"""Adapters turn a framework object into a feature function raw_rows -> {channel: array}."""
from __future__ import annotations

import os
from typing import Callable, Dict

from ..replay import load_file_module


def _resolve(ref: dict, paths: Dict[str, str], root: str):
    if not ref or not ref.get("source_file_id") or ref["source_file_id"] not in paths:
        raise LookupError("the recorded object has no usable source file; pass fn= to replay")
    path = os.path.realpath(paths[ref["source_file_id"]])
    if not path.startswith(os.path.realpath(root) + os.sep):  # re-checked on the path actually imported
        raise LookupError("recorded source resolves outside the chosen root")
    obj = load_file_module(path, fresh=True)
    for part in ref["qualname"].split("."):
        obj = getattr(obj, part)
    return obj


def build_target_fn(target: dict, paths: Dict[str, str], root: str) -> Callable:
    """Rebuild the feature function from the recorded source files (hash-verified by the caller before this runs)."""
    kind = target.get("kind")
    if kind == "callable":
        return _resolve(target["fn"], paths, root)
    if kind == "backtesting.py":
        from .backtesting_py import make_feature_fn
        prep = _resolve(target["prepare"], paths, root) if target.get("prepare") else None
        return make_feature_fn(_resolve(target["strategy"], paths, root), prep, target.get("params") or {},
                               target.get("backtest_kwargs") or {})
    raise ValueError(f"unknown target kind {kind!r}")

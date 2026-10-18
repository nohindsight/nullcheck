"""Display-only evidence: no decisions, comparisons, coverage or reducer changes."""
from __future__ import annotations

import json
import pandas as pd


def token_display(token):
    """Decode only canonical, supported builtins; never invoke user object reprs."""
    if token is None:
        return "missing"
    if token.startswith(b"time:"):
        _, zone, nanos = token.split(b":", 2)
        return pd.Timestamp(int(nanos), tz="UTC" if zone == b"aware" else None).isoformat()
    if token.startswith(b"date:"):
        return token[5:].decode("ascii")
    if token.startswith(b"str:"):
        # JSON escaping keeps newlines, control characters and lone surrogates safe.
        return json.dumps(token[4:].decode("utf-8", errors="surrogatepass"), ensure_ascii=True)
    if token.startswith(b"bytes:"):
        return repr(token[6:])
    if token.startswith(b"bool:"):
        return "True" if token == b"bool:1" else "False"
    if token.startswith(b"number:"):
        numerator, denominator = map(int, token[7:].split(b"/"))
        return str(numerator) if denominator == 1 else repr(numerator / denominator)
    return "value unavailable"


def finding(diff, prefix_length, categorical, fn, raw):
    """Retain comparison fingerprints and add readable values and a display rank."""
    result = dict(diff, prefix_length=prefix_length)
    channel = diff["channel"]
    layouts = getattr(fn, "_nullcheck_layouts", {})
    state = any(mode in {"scalar", "state"} and
                (channel == "attr:" + name or channel.startswith(("attr:" + name + "[", "attr:" + name + ".")))
                for name, mode in layouts.items())
    result["display_priority"] = (2 if state else 0) + int(categorical)
    result["categorical"] = categorical
    codec = getattr(fn, "_nullcheck_codec", None)
    tokens = []
    for side in ("full", "prefix"):
        value = diff[side + "_value"]
        token = codec.seen.get(int(value)) if categorical and codec is not None and value is not None else None
        tokens.append(token)
        result[side + "_display"] = (token_display(token) if token is not None or value is None
                                      else "categorical value unavailable" if categorical else repr(value))
    if categorical and state and all(token is not None and token.startswith(b"time:") for token in tokens):
        from .categorical import canonical
        try:
            endpoints = [canonical(raw.index[-1]), canonical(raw.index[prefix_length - 1])]
        except (TypeError, ValueError, OverflowError):
            endpoints = []  # An unsupported index type must never affect detection.
        if tokens == endpoints:
            result["hint"] = ("Init state matches each run's last row timestamp (data-end bookkeeping?). "
                              "Trade use is not assessed.")
    return result


def earliest_findings(res):
    """One earliest observed example per channel; never hide secondary channels."""
    candidates = res.findings or [dict(p["first_divergence"], prefix_length=p["prefix_length"])
                                 for p in res.probes if p["first_divergence"]]
    earliest = {}
    for item in candidates:
        old = earliest.get(item["channel"])
        if old is None or (item["row"], item["prefix_length"]) < (old["row"], old["prefix_length"]):
            earliest[item["channel"]] = item
    return sorted(earliest.values(), key=lambda item: (item.get("display_priority", 0), item["row"],
                                                       item["prefix_length"], item["channel"]))

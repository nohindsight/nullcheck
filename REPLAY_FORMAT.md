# Replay format 0.4.0 (nullcheck 0.2.4)

New records declare `contract.categorical_channels` and the required encoding
`BLAKE2B_53_CANONICAL_V1_EXACT_COLLISION_CHECKED`. These channel values are exact
integer fingerprints, not magnitudes: finite values diverge on inequality regardless
of numeric `atol`/`rtol`. None, NaT, pandas NA and numeric NaN encode as NaN and retain
the existing missingness/skip rules. Canonical tags distinguish strings, bytes,
booleans, dates, timestamps and numbers; equivalent integer/float labels share a token,
and timestamps use nanoseconds with distinct naive/aware tags (aware timestamps normalize to UTC). Unsupported values and fingerprint collisions
are unassessed. A codec checks collisions across the full/prefix comparison context;
reducer contexts start fresh and repeat their full/short comparison.

This explicit contract prevents an older numeric validator from reinterpreting a
categorical finding under numeric tolerance. Categorical fingerprints alone never
satisfy minimum meaningful coverage. Replay compares encoding channel lists as well
as findings, and revalidates reduced categorical cases. Human-readable categorical
values are not added to records; reports identify the codes as fingerprints.

The 0.2.0 and 0.3.0 schemas remain byte-for-byte available for validation. Exact replay
of those records requires the original runner and returns UNSUPPORTED_RUNNER before
user code executes. The 0.4.0 schema is packaged alongside them.

## Historical 0.3.0 semantics (retained except categorical exclusions)


This is a strictly versioned extension of REPLAY_FORMAT 0.2.0. File identity, paths, privacy, hashing, strict JSON parsing, modified-replay opt-in, source loading, tolerance, and reducer verification retain their prior meaning. Its historical schemas remain packaged unchanged.

## Why the bump is necessary

0.2.0 requires `DECLARED_AND_LEADING_BOTH_NAN`: one contiguous compared range after declared/leading warm-up; any unassessed probe forces NOT_ASSESSED. The required v0.2 fixes audit warm-up values, allow sparse matching-NaN gaps, and retain numeric comparisons when another channel is unavailable. Emitting these under the old constant would misstate the predicate and violate its validator. The new format avoids silent reinterpretation.

## New semantics

- `contract.warmup_policy = AUDIT_ALL_ROWS_SKIP_MATCHING_NAN`. `warmup_rows` records the user hint but does not exclude rows. `terminal_excluded_rows` is unchanged.
- `contract.repeatability` also allows PARTIAL: at least one baseline channel was excluded, while stable channels remain eligible for findings.
- `probes[].state` also allows SKIPPED_BOTH_NAN. Such a probe has no compared ranges or findings, a nonempty reason, and only matching-NaN skipped ranges covering every channel. It is never numeric coverage.
- COMPARED probes allow multiple disjoint compared ranges. Every declared channel is partitioned exactly into compared and skipped [start,stop) intervals over [0,L). Supported skips: leading matching NaN, other matching NaN, invalid numeric cells, whole-channel unavailability, and declared terminal exclusion. Arbitrary omissions are invalid.
- A COMPARED probe may have a reason describing unavailable channels/cells. All whole-channel skips and invalid numeric skips require that reason. Missing/unstable/non-numeric channels are identified by name.
- Status precedence: ERROR, then any stable observed DIVERGENCE, then NOT_ASSESSED if any unavailable probe/channel/cell, failed/partial repeatability, missing raw-prefix recomputation, zero numeric coverage, or insufficient meaningful coverage; otherwise NO_DIVERGENCE_OBSERVED. Matching-NaN-only skips do not force the whole result unassessed if other numeric coverage exists.
- Planning audits short prefixes even when warmup_rows is set. Data-aware candidates include NaN/zero/stale-run boundaries, robust price changepoints, and observed month-ends, subject to the deterministic 128 cap. Config plus input reproduce the exact plan.

## Legacy records

The 0.2.0 JSON schema is retained byte-for-byte. `validate` dispatches to its original range/status/planner semantics and hash/path checks. `replay` validates a legacy record, then returns `UNSUPPORTED_RUNNER` before importing user code: use the original pinned nullcheck 0.1.x runner to reproduce its original result. Legacy files are not rewritten, upgraded or executed with a different predicate. Unknown schemas are refused.

## Verification

Tests cover invalid skip reasons, range partitions, no vacuous numeric success, per-channel exclusions, sparse/long warm-up replay, missingness inside declared warm-up, scalar/list/private state, and unchanged strict-loader/path/hash regressions. Existing 0.2.0 records are checked during builder verification as an additional compatibility smoke test.

## v0.2.1 implementation clarification

The format remains 0.3.0. Attribute layouts are fixed per full-prefix comparison context; reducer predicates establish a fresh context on their own long input. Unsupported instance attributes are named by type in limitations and receive unassessed channels. Temporal-column refusals reuse NOT_ASSESSED without changing schema. CSV replay restores recorded datetime64 and Period column dtypes before checking order or executing the callback.

## v0.2.2 implementation clarification

Schema remains 0.3.0. External dependencies use named `input:*` coverage channels with no compared ranges and whole-prefix NOT_ASSESSED skips. They set raw_prefix_recomputed=false, preventing a clean result even if feature values are equal. The input channels include dependencies observed only in shorter prefixes. A file-dependent record is valid but reproduction is UNAVAILABLE: external data was neither copied nor hashed. In v0.2.2–v0.2.3 categorical adapter values were excluded; v0.2.4 replaces that behavior with equality comparisons, including prepared columns.

Real-world integration also records `SNAPSHOT_SCOPE=INIT_STATE_ONLY` in contract limitations when an adapter run captures no numeric row-wise channels. Starting in v0.2.3, that scope alone cannot qualify numerical agreement. In v0.2.2–v0.2.3 metadata added by prepare() was excluded per column; v0.2.4 compares those values; numeric/mixed frames still require the exact input index. No schema relaxation was used to accommodate the 300x4 state reproduction: the pre-broadcast cap produces one unavailable channel and a valid record instead.

## v0.2.3 minimum coverage and replay

The schema remains 0.3.0. A compared probe may carry the reason `insufficient meaningful coverage` while preserving all its compared and skipped ranges. Existing status precedence therefore enforces NOT_ASSESSED without fabricating skipped channels or falsely changing `raw_prefix_recomputed`. The report lists the channels excluded by the minimum and why; a divergence is preserved even if all other channels are trivial.

The runner reserves two additional callback runs for a repeated numeric-input dependency probe; `max_callback_runs = 4 + 2 * planned_prefix_count + reduce_budget`. Results of that artificial input are not prefix probes or leak findings. Candidate channels must contain compared variation beyond constants/raw copies/affine counters and respond on their originally compared rows. Sparse finite-event disappearance under the perturbation counts as response; matching NaNs still do not count as numeric agreement.

Records from runners 0.2.0–0.2.2 still validate with their original callback budget. Exact replay returns UNSUPPORTED_RUNNER with a request for the original version because the coverage semantics changed. Frozen historical packages and schemas are unchanged.

## v0.2.4 dependency and shape details

The existing two reversal runs now also exclude channels commuting with the numeric-row permutation. This changes no callback budget or prefix ranges. Bounded state layouts expose a categorical `[shape]` channel in addition to elements; a dimension change can be a divergence even if individual elements are unassessed. Shape codes use the same exact categorical contract.

## v0.2.5 presentation supplement

`findings.json` is an additional private presentation file: one earliest differing row per channel and tested prefix, with numeric fingerprints, decoded full/prefix values and display priority. It preserves secondary channels that are not selected as a probe's `first_divergence`. Like `report.txt`, it is not schema-validated, hash-bound or independently certified by `validate`; editing it does not affect comparison replay. The authoritative schema 0.4.0 record and its comparison/reducer logic are unchanged. Treat readable values as private strategy/data content.

Default-argument containers now use `input:default:` channels and default-state reasons. They remain NOT_ASSESSED with exact replay unavailable; runtime state was not captured. Exact label matching against old records may require their original runner.

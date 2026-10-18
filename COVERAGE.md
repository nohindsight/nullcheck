# What nullcheck 0.2.5 checks

For each selected prefix length L, the callback receives a fresh deep copy of raw rows [0,L).
The check compares stable numeric channels with their full-history values, over [0,L-terminal_excluded).
Declared warm-up never masks a changed value or missingness.

- Two finite values use the symmetric tolerance `abs(a-b) <= atol + rtol*max(abs(a),abs(b))`.
- Exactly one finite value is a MISSINGNESS divergence.
- Matching NaNs are skipped, including warm-up and sparse signal gaps; they do not count as numeric agreement.
- Inf pairs and finite magnitude above 1e300 are unassessed. An invalid cell/channel cannot erase a stable divergence elsewhere.
- Each input index must be unique, non-missing and increasing. Every datetime64/Period column is checked too, as are ISO date strings and YYYYMMDD strings/integers (including CSV `--no-dates`). Numeric/string epoch keys whose names have explicit date/time/epoch tokens or ts/unix aliases (including camel-case boundaries) are checked for strict increasing order without guessing their unit. Ambiguous named date/time columns are refused rather than guessed. Numeric/non-temporal indexes disclose that index chronology is unverified. No sorting, input coercion, price adjustment or filling is performed by the checker.

DIVERGENCE_OBSERVED means prefix dependence in the observed snapshot. It does not establish that dependence is unintended, used for trading, or economically material. Init scalars/lists are treated as state available throughout the input: e.g. a stored full row count or a deliberate training label can flag. Inspect the named channel and its use.

NO_DIVERGENCE_OBSERVED requires meaningful coverage from at least one non-trivial compared channel with observed numeric-input and cross-row dependency, no observed divergence, no unavailable probe/channel, and raw-prefix recomputation. Probes containing only matching NaNs may be skipped; all-missing channels have no numeric coverage and are disclosed. All-missing output alone is NOT_ASSESSED. Replay records partition each channel into compared and skipped ranges, and rendered reports state skipped prefixes and unassessed channels.

## Limits

1. Columns computed before the callback, stable caches, other symbols/files/network reads, revised data, survivorship and back-adjustments are not made point-in-time by truncation. Generic callbacks must receive raw rows. The backtesting adapter marks precomputed non-OHLCV columns NOT_ASSESSED.
2. Global/class/cache/random state is not reset; two repeats only detect observed instability. Object internals (e.g. fitted model objects), dictionaries and nested non-numeric state are not traversed. Unsupported instance attributes are named with type and produce an unassessed channel rather than being silently dropped.
3. `next()`, trades, orders, fills, portfolio value and selection bias remain outside scope. Only `Strategy.init()` is snapshotted. Framework `_broker`, `_data`, `_params`, `_indicators` containers are not treated as user feature attributes.
4. Prefixes are finite samples. The 128 cap can omit anomalies and month-ends in long datasets. Non-anomalous rare rows can be missed; increasing sampling is not proof of no leak. `nan_cutoffs=False` disables data-aware additions.
5. Generic string/categorical outputs remain unsupported. The backtesting adapter compares supported strings, categories and datetime values by exact equality, using collision-checked canonical fingerprints. None/NaT/NA/NaN are missing; constant metadata agrees and changing metadata can diverge. Arbitrary object reprs are never used. Fingerprints have no numeric ordering or magnitude and cannot alone establish minimum meaningful coverage. Booleans retain their numeric 0/1 encoding; missing boolean arrays remain unsupported.
6. Terminal exclusion is still an explicit loss of coverage and can hide short-horizon leaks. Differences below tolerance are not flagged.
7. No causal-availability, trade-effect, profitability or general recall guarantee is made. Validation counts describe the tested synthetic population, not unknown user code.

## Supported inputs and adapter

Generic `raw_rows -> Series | DataFrame | ndarray | dict` with one value per row. Pandas outputs retain the same index and order. Numeric lists in generic output retain the row-aligned contract.

backtesting.py 0.6.6 only. Indicators and row-aligned ndarray/Series/DataFrame attributes (including private names) are snapshotted. Numeric scalars (including NumPy booleans), list/tuple state and non-row-length ndarray/Series/DataFrame state are compared as init-state snapshots, one broadcast channel per element. Row-aligned matrices may use the first or last axis (the last axis wins when the full-run matrix is square). Layout is inferred from the initial full run and fixed across prefixes; state dimensions also receive exact categorical shape channels, so changed dimensions can diverge; impossible element layouts remain named unassessed channels. Lists mean state vectors; use ndarray/Series for row-aligned attributes. Empty arrays/zero-width matrices, complex values and nested non-numeric state are individually unassessed; datetime values use exact equality. Attribute conversion failures are isolated so other numeric channels still run. Arbitrary object internals remain outside scope, with named exclusions. Each reducer candidate uses a fresh layout context and must re-trigger independently. The adapter depends on private upstream APIs and refuses other versions.

## Validation and migration

`validate.py` retains the original 30 seeds x 8 leak / 8 clean synthetic fixtures; the full matrix was last run for v0.2.4 (pre-release). v0.2.5 ran the regression suite and five disclosed real-world cases only. Regression tests derive from disclosed minimal repros from an internal blind test suite and from a real-world trial that included the TrailingStrategy back-fill. Held-out case files were not inspected.

The original NextMutation fixture stores `_n=len(data)` in init; v0.2 flags that real scalar prefix dependence while its trace stays causal. A separate NextOnlyMutation control verifies that next() is never run. This is an intentional expansion of snapshot coverage, not a new claim about trade outcomes.

New records use schema 0.4.0; see REPLAY_FORMAT.md for the strict semantic change and legacy validation policy.

The v0.2.1 regressions derive from an internal review (B1/B2) and the R4b quantile form described in the blind suite's v0.2.0 report, read only after that evaluation was complete. No held-out case files of the later round were accessed. R4b is a disclosed regression, not held-out validation.

## v0.2.2 bounded state and external-input coverage

State expansion is capped before array conversion/broadcast at 256 elements and 1,000,000 broadcast cells per attribute. A larger list/tuple/tensor receives one named NOT_ASSESSED channel with the cap reason; other attributes still compare. Lists/tuples of row arrays should be returned as a row-aligned ndarray instead. This bounds checker expansion, not memory/time consumed by user callbacks.

The checker observes read-open events during callback/prepare/init execution and inventories directly referenced pre-existing arrays, pandas objects, containers and opaque objects in user globals, closures, defaults and class state. It follows a bounded graph of user helper functions. Each observed input has a separate `input:file:...`, `input:memory:...`, `input:object:...` or `input:default:...` coverage channel, skipped as NOT_ASSESSED at every prefix. Even identical fixed predictions cannot support a NO_DIVERGENCE result. Findings in other channels retain DIVERGENCE precedence. An observed dependency is conservative: a configuration container may be marked even when harmless.

Observed external files are **not opened again, copied or hashed by nullcheck**. Paths and dependency names appear only in the local private report. Such records validate but have replay availability UNAVAILABLE because input identity is not captured. Source/import reads are excluded only when identified as interpreter loading. Observation uses a context-local Python audit hook; outside the callback the hook is inert.

This is not a sandbox or complete dependency proof. Undeclared network/native-extension reads, reads in other threads/processes, dynamically resolved dependencies, precomputed scalar constants and code loaded before observation may evade inventory. Absence of observed external inputs does not establish their absence. Truncation cannot establish point-in-time availability of an external input. Do not run code you do not trust.

### Disclosed real-world metadata and scope

Categorical comparisons also cover datetime metadata attributes and metadata columns added by prepare(), including mixed row-aligned DataFrames. Their names remain visible. Wrong DataFrame indexes are still NOT_ASSESSED; value comparison never substitutes for index equality. Empty list/tuple/dict/set/frozenset metadata is excluded only while empty; a populated prefix becomes NOT_ASSESSED. Non-empty dicts, opaque objects and unsupported registered indicators still cannot support a clean result.

Backtesting reports with only scalar/vector init state carry `SNAPSHOT_SCOPE=INIT_STATE_ONLY` and now return NOT_ASSESSED with reason `insufficient meaningful coverage`. A few constant counters are not presented as an assessment of next() or trading logic. External predictions and side inputs remain individually NOT_ASSESSED even when their other channels agree.

The 256-element state cap also prevents the disclosed 300x4 numeric lookup-table expansion from exceeding the replay schema's 1000-channel limit. The cap is per attribute; the replay schema still limits total channels to 1000. This is an engineering bound, not a claim of complete coverage for arbitrarily wide feature sets.

## v0.2.4 minimum meaningful coverage

Constants on compared cells, copies of raw columns/index, and affine row counters do not qualify. At least two compared finite rows are needed; the screen never borrows variation from an untested tail. Sparse fixed-amplitude signals with interior gaps may qualify by their event locations, but warm-up alone does not count. Their matching NaNs remain explicitly outside numeric coverage.

Remaining candidates must respond at previously compared rows when complete numeric input rows are jointly reversed, preserving index/time keys, dtypes, missingness and cross-column relationships. The same runs compare each response with the correspondingly permuted full output at originally compared rows. Channels commuting with that permutation (including log(Close), Close/Open, hl2 and current-row event flags) cannot qualify. The permuted full output may include warm-up NaNs; a mismatch there can evidence cross-row dependency. This deterministic dependency probe runs twice and uses at most two additional calls. It checks dependency, not prefix divergence: its errors or absent response are NOT_ASSESSED, not a leak finding. Numeric values in the caller's input are not changed. Unsupported channels, external-input restrictions and divergence precedence still apply independently.

This deliberately conservative rule can leave causal but constant-looking or counter-looking indicators unassessed, especially with short/linear data. Time-only/non-numeric features and features invariant or equivariant to this perturbation may also lack established dependency. The index and time keys stay fixed, so this is a bounded heuristic, not a mathematical characterization of all row-local functions. It does not establish arbitrary provenance, prove point-in-time availability, or prove a feature is traded. The fixed-seed noise control remains a causal non-leak but is now NOT_ASSESSED because it does not depend on raw values; it accounts for 30 of the 240 original clean fixture runs.

Categorical values use schema 0.4.0 exact equality regardless of atol/rtol, with deterministic 53-bit BLAKE2b fingerprints checked for collisions across full/prefix runs. Unrepresentable timestamps or unsupported mixed objects are explicitly unassessed. State dimensions have their own exact equality channel; changes can diverge even where element channels are unavailable. Caps are applied before shape/value expansion.

An object referenced only through discarded calls to inspectably inert Python methods is excluded from external-input inventory. The exception requires ordinary attribute lookup and method bytecode that only returns None, such as a literal `pass` progress shim. Real method bodies, property/data reads, consumed return values and uninspectable calls remain conservative inputs. This is not a general logger/progress-object exemption.

## v0.2.5 presentation and known limitations

**True but untraded:** nullcheck reports that future information is AVAILABLE to the strategy in the observed init snapshot, not that it was traded. A last-dataset timestamp used only for printing diagnostics, a training label or a plotted value can be a real divergence without affecting orders. `next()` and trade use are not checked. Inspect how the named value is used.

Reports show the earliest observed differing row for **every divergent channel**, both readable values and the prefix length. Numeric row indicators/signals take headline priority over categorical row channels, then numeric init state, then categorical init metadata; rows and prefix lengths break ties within a group. This is a presentation heuristic, not proof of trading importance. The original per-probe `first_divergence`, comparison counts and reducer selection are unchanged; a small reproducer can therefore concern a different channel from the headline. A timestamp matching each run's last index gets a tentative “data-end bookkeeping?” hint without suppressing the finding.

The private `findings.json` supplement lists one first differing row per channel and prefix, retaining numeric comparison fingerprints alongside readable labels. `report.txt` shows one earliest example per channel. Neither presentation file is independently certified by the standalone record validator; `replay.json` remains the schema 0.4.0 comparison record. Time values are rendered in ISO form; timezone-aware timestamps use UTC. Missing values are displayed as `missing`. Literal strings are quoted and escaped.

Default-argument containers and opaque objects are classified as **default-argument state**, not automatically as external data. They remain conservatively NOT_ASSESSED because their runtime contents and recomputation are unverified; this label change does not turn a mutable default into assessed coverage. Its record remains unavailable for exact replay. Genuine file/global/closure dependencies keep their external-input classification. Exact comparison of older default-state records may require the original runner because channel/reason labels have changed.

Known limits, unchanged in this release:

- **K4r — index/position mixing can overstate meaningful coverage.** Row-local expressions such as `Close * dayofweek` or `Close * arange(n)` can qualify because the reversal keeps the index fixed. They cannot themselves read future rows, and other channels are still compared, but qualifying is not proof of a past-dependent indicator.
- **K5 — next-only inputs are missed.** Precomputed predictions read only in `next()` can coexist with a NO_DIVERGENCE result. nullcheck stops after `init()`; it does not check `next()` or establish whether a signal was traded.
- **K8 — validation does not certify meaningful coverage.** The standalone record validator checks structural/semantic consistency, not the dependency-screen conclusion. Replay reruns the screen.
- **K10 — large tuples of row arrays remain unassessed.** State expansion caps can leave a tuple containing a leak NOT_ASSESSED. Use an appropriate row-aligned ndarray layout; no complete detection claim applies to capped state.
- **K11 — mixed object numbers use exact equality.** Lists or object Series mixing numbers with None/strings use categorical equality, ignore numeric tolerances and cannot establish meaningful numeric coverage. Tiny order-sensitive floating-point differences can therefore flag; no such false alarm was observed in the reviewed controls.

The disclosed D01 RangeIndex DataFrame remains NOT_ASSESSED when its index differs from the raw time index. Positional comparison is outside this presentation release. No new detection coverage is claimed.

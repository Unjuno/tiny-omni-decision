# R2 crop × coverage factorial comparison

## Purpose

The full-extent, unique-coverage arm will complete a fixed 2×2 comparison of
visual feature extent (`center`, `full`) and sample policy (`repeat`, `fresh`).
This separates the two main effects from their interaction instead of treating
the newest arm as a stand-alone quality result. All four cells use the same
ordered 24,346-row development validation set and the fixed final optimizer
step (32,768) as the primary comparison. Each run's selected best step is
recorded as context but is not used for the primary factorial estimate.

The analyzer is CPU-only. It loads saved prediction arrays; it does not load
models, fit parameters, evaluate the sealed audit set, or change any run.

## Implementation and safeguards

`scripts/analyze_native_coverage_factorial.py` checks each run's expected status,
endpoint step, explicit `sealed_audit_loaded: false` field, exact validation ID
order against the frozen common corpus, and SHA-256 entries for the IDs and
endpoint prediction array. It refuses to calculate effects if a cell is
missing or invalid. It reports per-cell Accuracy/NLL/Brier/ECE and macro
metrics, four simple effects, and the difference-in-differences interaction:

```text
(full_fresh - full_repeat) - (center_fresh - center_repeat)
```

The paired bootstrap resamples `group` clusters within each modality and reuses
the same draws across the four conditions. It reports 95% percentile intervals
for Accuracy, NLL and Brier contrasts using seed 17 and 10,000 resamples. ECE is
reported as a point estimate because it is nonlinear and not represented as a
row-additive bootstrap contrast in this implementation. The endpoint is the
predeclared primary comparison; selecting different checkpoints against this
same validation would add checkpoint-selection bias.

## Current preflight

Preflight artifact:
`C:/CodexArtifacts/pretrained-reuse-r2/native-coverage-factorial-preflight-02/preflight.json`

Result SHA-256: `138cd5cf16f9cee6e89cc118ad3f636b34cd467d26023a2d79fc038ed8ed5725`
Analyzer commit: `18c5284`; analyzer SHA-256:
`a8cc68935d49638e8d499ff514c3841ceae7bc9ab8fe2a8c2fa2f9672434b166`.

The three completed inputs passed integrity checks. Each uses the exact common
validation ID hash
`db7d0f1d09319a48a7725ed36ccb515cdcc85a409f019400cf0062be38768f30` and has
24,346 rows:

| Cell | Fit artifact | Result SHA-256 | Endpoint predictions SHA-256 | Best step |
|---|---|---|---|---:|
| center × repeat | `native-long-budget-fit-01` | `645bedbe73ed508fad68c4f636a41df6c8630cd996641c44ce2a87d31550250b` | `78dcb283c190bcb6a09e2eda2aaecee4b8532124ead2a053a9d1bb61743d7629` | 8,192 |
| center × fresh | `native-coverage-long-fresh-fit-01` | `c93da50addd7e92760cfe1878b126c2fcf8953612d8afd2d106d86d9cec779bf` | `ded2ebec114edab0aefb54baa6c54c2de42608d41aa3c4fb0ebed454ae483005` | 8,192 |
| full × repeat | `native-full-extent-fit-01` | `f87f4194992a22d6b3bd2095d725725c6debfa35188c1c9b585ee6b0fbaf0a83` | `e03f602988bfbf357e0f98f06a806e2de40df1fa6a07bd0dcbf53c660ee87a12` | 12,288 |
| full × fresh | `native-full-extent-fresh-long-fit-01` | missing | missing | not run |

Preflight status is `BLOCKED`, with only the full × fresh cell missing. No
factorial effect or quality estimate was produced. That arm is the separately
preflighted long run from
[`R2_FULL_EXTENT_UNIQUE_COVERAGE_LONG_20261025.md`](R2_FULL_EXTENT_UNIQUE_COVERAGE_LONG_20261025.md).
Its GPU smoke and fit were not started because unrelated live PID 26920 owned
the RTX 3080 Laptop at the time of this preflight. No process was interrupted.
No sealed audit was loaded, and no existing run files were changed.

## Verification

- Added CPU tests cover simple effects/interaction, missing and misaligned
  conditions, deterministic paired group bootstrap, independent-group minimum,
  artifact pin validation, fail-closed preflight, and complete output writing.
- `python -m pytest tests/test_native_coverage_factorial.py tests/test_native_full_extent_analysis.py tests/test_native_full_extent.py -q`: 28 passed.
- Full CPU suite: `python -m pytest -q`: 2,397 passed, 11 skipped.
- `ruff check .` and all CI manifest/catalog validation commands: passed.
- `ruff check scripts/analyze_native_coverage_factorial.py tests/test_native_coverage_factorial.py`: passed.
- Actual artifact preflight: 3 cells passed, 1 missing; no quality metrics.

When the fourth fit and its validation endpoint prediction exist, rerun with a
new output directory:

```powershell
python -m scripts.analyze_native_coverage_factorial `
  --output C:/CodexArtifacts/pretrained-reuse-r2/native-coverage-factorial-comparison-01 `
  --step 32768 --seed 17 --resamples 10000
```

The 2×2 result is a development-validation comparison, not a blind evaluation,
general video claim, or product promotion. Even the fresh-coverage cell adds no
new Video questions, so this comparison does not establish improved Video data
coverage.

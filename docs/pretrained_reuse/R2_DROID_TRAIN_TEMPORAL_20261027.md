# DROID train-only temporal and target diagnostic

## Finding and decision

The existing DROID next-action task has strong **training-set command
persistence**: copying the last observed commanded direction predicts the next
command in 260/275 episodes (94.55%). This uses an additional action-history
input that the previous VideoPrism/sensor readouts did not receive. It is a
task-autocorrelation diagnostic, not model quality, a fair comparison to those
readouts, or evidence that the 95–98% product goal has been reached.

A fixed direction rule on the original sparse positions reaches 46.18%; using
the immediately preceding position instead reaches 53.45%. This motivates
testing temporal representation before adding fusion capacity. It does not
prove a representation bottleneck, generalization, or a benefit from video.
Keep R2 and the four-modality quality gates open. UnAV-100, including its
annotations/features, remains excluded pending rights resolution as requested.

## Frozen scope and predeclared rules

Use exactly the 275 train rows from the existing fresh-shard experiment on
`codex/r2-droid-fresh-holdout-20261026`, commit
`eba818067c6a9ef9089288205c26f6a8254ca6e5`. The earlier results and original
train/validation split are unchanged. Each selected row has a distinct recording
group. No validation manifest or predictions were opened, selected or scored.
No sealed audit, fitting, encoder, checkpoint or GPU ran.

The new script and its saved protocol fixed these four rules before the source
records were processed. There was no threshold, lag, prompt or model search:

1. Majority train label (`x+`), computed from these train targets.
2. Previous command: dominant signed XYZ command at `k`, predicting the existing
   target at `k+1`. **Additional input** relative to the old readouts.
3. Sparse displacement: XYZ position at `k` minus position at the immediately
   preceding original input index. Uses data already present in the old sensor
   input, with a different deterministic readout.
4. Dense displacement: position at `k` minus position at `k-1`.
   **Additional past observation** when `k-1` was not originally sampled.

Directions use the original seven options `stop, x-, x+, y-, y+, z-, z+`.
Maximum absolute XYZ <= `1e-4` maps to stop; ties use the first axis in XYZ
order. For displacements this is a fixed threshold in native displacement
units, not a velocity threshold. No physical-unit or coordinate-frame equivalence
between command and observation is established. No division by variable index
gaps or post-result threshold adjustment was performed.

All inputs to the rules are at or before `k`. Only the train target consistency
check reads `k+1`. Tests change future actions and positions and require
identical predictions. Explicit per-step physical timestamps are unavailable;
index gaps must not be reported as measured seconds.

## Executed results, train only

| Fixed rule | Correct / 275 | Accuracy | Mean per-class recall |
|---|---:|---:|---:|
| Train majority | 102 | 37.09% | 14.29% |
| Previous command, extra action history | 260 | 94.55% | 90.02% |
| Original sparse-position displacement | 127 | 46.18% | 34.82% |
| Dense-position displacement, extra past sample | 147 | 53.45% | 45.86% |

All seven classes are present. Train target counts are unchanged:
`29, 8, 102, 18, 23, 71, 24`. The previous-command rule still misses 3/8
`x-` targets. Class imbalance and the small minority counts remain relevant.
The machine-readable receipt retains every confusion matrix and per-class
count/recall. NLL, Brier, ECE and probability calibration are not defined for
these deterministic rules; no artificial probabilities were assigned.

The last sparse observation gap ranges from 1 to 120 source steps, median 8.
Dense displacement gets 37 episodes correct that sparse displacement misses;
sparse gets 17 correct that dense misses. The net improvement is 20 train
episodes (+7.27 percentage points); this is exploratory training evidence, not
a validation gain or a blind paired-effect estimate. Changing the interval also
changes displacement magnitude relative to the fixed stop threshold.

All 275 targets recomputed from original commands matched the frozen train
labels exactly. Nine original TFRecord shard SHA-256s matched the pinned
download receipt, totaling 8,567,919,011 bytes. All 428 opaque records were read
sequentially and their source identities parsed; 153 nonselected records were
skipped without extracting their numeric features/targets. This is not a claim
that no validation bytes entered memory. No image/video payload was decoded.
The reused minimal parser checks record framing, not TFRecord CRC32C; exact
whole-shard SHA-256 equality supplies integrity relative to the saved receipt.

## Reproducibility and protected artifacts

New output, outside Git:
`C:/CodexArtifacts/pretrained-reuse-r2/droid-train-temporal-baselines-20261027-attempt-01/`.
It contains `protocol.json`, the pinned parser, `train-predictions.jsonl` and
`result.json`. The existing data, prep manifests, features and model artifacts
were only read. Receipt/train hashes were checked again after execution.

- Base worktree commit: `214726925cbee9fb12c2a058ff5c520df2b59723`.
- Execution: 2026-10-26 16:07:12–16:07:32 UTC (October 27 JST).
- CPU body time including shard hashes/parsing: 20.27 seconds; startup excluded.
- Environment: Windows build 26300, Python 3.11.9, NumPy 2.4.6; no upgrades.
- No model, GPU, network data acquisition, or cloud spend; RAM peak unmeasured.
- Train SHA-256: `4e14850bbbe2eee2a96116a97ca935dfd5dea91d197c00639ea7ed02e2d93341`.
- Receipt SHA-256: `5d5a5ba5d59b8c76790db1c9399f6490e70098ad390efafe5ef7f94df63ee6f7`.
- Executed script SHA-256: `13147a37964662c11c449a10b7a6c619fdf928aad523d8c78e2e9608f6068774`.
- Pinned parser SHA-256: `1d5d7da7bba74c5b4ebc13636efbbd9cce2c510a90509ac744fefc268fb5b659`.
- Per-row predictions SHA-256: `6cc30b04d1d10d880a157885102848ac882a88ad8580bd17fb9411744cb309ae`.
- [Result receipt](R2_DROID_TRAIN_TEMPORAL_20261027.json) SHA-256:
  `d4f8c1e2db7a98088dcb9924e389a703703695114e25e6a5a80777dbc6815bb6`.

```powershell
python -m scripts.audit_droid_train_temporal `
  --data-root C:/CodexArtifacts/pretrained-reuse-r2/droid-fresh-next-action-seed17-attempt-01 `
  --train C:/CodexArtifacts/pretrained-reuse-r2/droid-fresh-next-action-prep-attempt-02/train.jsonl `
  --output C:/CodexArtifacts/pretrained-reuse-r2/droid-train-temporal-baselines-NEW
```

The parser is obtained from the exact repository commit above; it is not
downloaded from an unpinned external source. Existing output directories are
rejected. Tests cover label ties/sign/stop, future-input independence, distinct
sparse/dense history, invalid input indices, finite inputs and minority metrics.
Ruff and all CPU tests passed: **2,626 passed, 11 skipped**. All CI-equivalent
model/dataset manifests, four catalogs and the example policy audit passed.
These manifest checks do not grant use of REVIEW sources such as UnAV-100.

## Next warranted experiment and remaining limits

Before another fusion-capacity experiment, preregister a train-only grouped-CV
comparison of the original raw sensor representation and a small fixed temporal
delta representation. Keep the same train labels and groups; isolate any added
observation from a same-input readout change. Do not tune against the already
observed development validation. Action persistence can remain a task diagnostic
but must not be presented as visual understanding or an adopted extra input.

Even a successful temporal readout would not show that this one-step task
requires multiple modalities. Input intervention and a prospectively frozen
independent task remain necessary. This diagnostic supports neither adding a
larger recurrent module yet nor declaring that recurrent architectures cannot
help. It leaves the parameter/package budgets and all-modal quality objective
unchanged. No production promotion follows this result.

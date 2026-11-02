# HABIT transition readout results — 2026-11-02

## Scope and decision

This is a fixed, exploratory transition-lookahead task: predict the next
low-level robot action from eight pre-transition video frames and/or eight
pre-transition bimanual proprioception rows, ranking the supplied action
descriptions. The action descriptions use the already audited frozen MiniLM
encoder. The train/validation split is the seed-17 whole-task-group split in
the [preflight record](R2_HABIT_SUPERVISOR_TASK_PREFLIGHT_20261102.md). The
reserved groups were not loaded.

The result does **not** admit HABIT as product-quality evidence. The best
validation accuracy was 44.44%, and all five readouts selected only epochs 1–5
from a fixed 25-epoch budget. The recurrent video+state model tied the
options-only model on accuracy and had worse validation NLL. The mean-pooled
state-only readout had the lowest validation NLL by a very small margin. This
does not establish a useful temporal-fusion gain.

## Frozen data and feature extraction

- Dataset: `configinc/HABIT@bae97d251907743f8fb6db7e70c61d361df4dc1e`.
- Split: seed 17, whole task-details groups; 936 train windows / 144
  validation windows across 11 / 3 groups. The three validation groups have
  64, 48 and 32 windows. Split and media gates are documented in the
  preflight record.
- Video encoder: VideoPrism Base F16R288, revision
  `be719a406d563b66f0ac969e7c94bab8e997c81a`, 114,365,184 parameters;
  checkpoint SHA-256
  `61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29`.
- Text option encoder: `paraphrase-MiniLM-L3-v2@4ca70771034acceecb2e72475f72050fcdde4ddc`,
  17,389,824 parameters; weight SHA-256
  `cf1e4e2d420c664973037c3c73125d7a8fc69952495093ef8f50596f8943a433`.
- Extracted cache: 312 episode shards, `[8,768]` video features and `[8,42]`
  state features per example. Train/validation sample IDs were verified in
  order against the frozen audit. `reserved_loaded=false`.
- Extraction ran on `TFRT_CPU_0` for 1,819.217 s (30 min 19 s). The earlier
  three-window warm-forward mean was 1.480 s, which projected 1,598 s for 1,080
  windows; the full run took 221 s (13.8%) longer because media decoding and
  per-window preparation are included.
- Feature receipt SHA-256:
  `ee8245daab3658090d380c4a6d07b276c61076c905811aae8569b651c96b086d`.
  Feature config SHA-256:
  `0d7973e2f524fed26ab42db45c2da57a8f1b0b397cdb28f016a60fcbadaddda1`.

The frozen backbones plus the selected LSTM readout total approximately
131,865,115 unquantized parameters. No quantization was done. The two frozen
weight files alone total approximately 475 MB decimal, so this extraction
experiment does not meet the provisional 200 MB package goal.

## Readout experiment

One fixed seed (17), hidden size 64, batch 64, AdamW learning rate `1e-3`,
weight decay `1e-3`, and 25 epochs were used for every condition. Each condition
selected its best checkpoint by validation NLL. Encoders remained frozen; only
the small readout was trained. The readout ran on CPU because the RTX 3080
Laptop was already at 97% utilization and 15,821 / 16,384 MiB at run start
(Orbitune worker PID 28688). No new GPU process or cloud compute was used.

| Readout | Parameters | Selected epoch | Train accuracy at selected epoch* | Validation accuracy | NLL | Brier | ECE-15 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Options only | 24,987 | 3 | 56.41% | 44.44% | 1.1457 | 0.6456 | 0.0847 |
| Video mean + options | 74,139 | 2 | 70.94% | 33.33% | 1.1702 | 0.6712 | 0.1972 |
| State mean + options | 27,675 | 2 | 72.33% | 44.44% | 1.1428 | 0.6369 | 0.0681 |
| Video + state mean + options | 76,827 | 5 | 89.74% | 31.25% | 1.1906 | 0.6857 | 0.1546 |
| Video + state LSTM + options | 110,107 | 1 | 56.73% | 44.44% | 1.1525 | 0.6433 | 0.0627 |

`*` Training values are resubstitution metrics, not a fixed-example
generalization gap. At epoch 25, video+state mean reached 98.82% train accuracy
while validation accuracy/NLL were 31.94% / 1.3845. The LSTM reached 100%
train accuracy while validation accuracy/NLL were 35.42% / 1.7311. The
training curve therefore shows clear overfitting after the early selected
epochs. All 25 epochs ran for all five conditions; best-checkpoint selection
did not stop training.

The LSTM and options-only predictions had identical accuracy in each of the
three validation task groups. The paired task-group bootstrap used 10,000
replicates and gave an accuracy delta of 0.00 percentage points with a
percentile interval of [0.00, 0.00]. With only three groups this is descriptive,
not a stable population interval. Mean validation choice-set size was 3.56;
the mean per-example uniform-choice accuracy expectation is 32.96% (not a
measured score from the model run).

### Input controls

Controls were computed after model selection on the same 144 validation rows:

| Readout | Normal accuracy | Zero video | Shuffled video | Zero state | Shuffled state |
|---|---:|---:|---:|---:|---:|
| Video mean | 33.33% | 44.44% | 33.33% | 33.33% | 33.33% |
| State mean | 44.44% | 44.44% | 44.44% | 44.44% | 39.58% |
| Video + state mean | 31.25% | 24.31% | 33.33% | 32.64% | 30.56% |
| Video + state LSTM | 44.44% | 38.89% | 44.44% | 44.44% | 44.44% |

The controls do not provide a consistent, strong video or state dependence
signal. They are validation diagnostics and do not justify another capacity
increase on this split.

## Execution integrity and artifacts

Feature extraction used the exact pinned source, media, and frozen feature
cache; it did not access the reserved split. The readout used no GPU, did not
change source media or encoders, and did not use a second seed.

The first readout attempt stopped before training because episode shards have
different candidate counts. The loader was corrected to pad the option axis;
the second attempt loaded 936/144 rows and completed all 125 epoch records.
It wrote all five best checkpoints, predictions, history, and `results.json`,
then returned exit code 1 because the final status print passed `flush` to
`json.dumps`. The print call was fixed in source afterward. This reporting
exception occurred after the output files were written; an independent reload
check matched all five saved checkpoints to their stored validation
probabilities within `8.95e-8`. No model was rerun after observing validation
metrics.

External artifact directory:
`C:/CodexArtifacts/pretrained-reuse-r2/habit-task-feasibility-20261102-attempt-01/`

Feature extraction invoked `scripts/extract_habit_video_features.py` from the
worktree with the pinned attempt root, source metadata, audit report, option
cache, VideoPrism checkpoint/source, and a new
`features-v1/videoprism-cache-01` output directory. It ran in the existing
VideoPrism Python environment. That venv did not contain PyArrow, so the
already-installed Python 3.11 site-packages directory was appended at runtime;
no dependency was installed or upgraded. The extraction-time script SHA was
not captured. The later source change was limited to the resume-only shard key
check; current script SHA-256 is
`d3c79e8ea6113c447d29d0796de979bc83e66d547c3c3eeb1c2bcab8d2b33bee`.

The readout command was run from source commit `9111d9a8` with the new
experimental script in the worktree:

```powershell
python scripts/train_habit_option_readout.py `
  --feature-cache C:\CodexArtifacts\pretrained-reuse-r2\habit-task-feasibility-20261102-attempt-01\features-v1\videoprism-cache-01 `
  --output-dir C:\CodexArtifacts\pretrained-reuse-r2\habit-task-feasibility-20261102-attempt-01\readout-v1-attempt-02
```

The readout script SHA was not captured at launch. Its current SHA-256 is
`2d226525171bc14edcd12ccea345211b0eb54aed69c002c03edf6df00fb7b188`;
post-run edits only fix terminal status printing and Python/Torch environment
labels.

- Feature cache: `features-v1/videoprism-cache-01/`.
- Readout output: `readout-v1-attempt-02/`.
- Effective readout config SHA-256:
  `efb5552acbdbbc97685c1e502c51449ec815db7bdd89b679f8a8d90a5e65a819`.
- Results SHA-256:
  `49f18d28d503f8f0d66ac3cc0900fb4ff975ad9897a589ca6e62cb6de89a67bb`.
- Training history SHA-256:
  `83abf34bad2d8aa8f59011aab55dfb165fc5e3e48817dbfd14ce8bd18f48fc35`.
- Supplemental execution-receipt SHA-256:
  `4af8f2c5584c89ee17bb677dedd58e0f8f2a1ce760acb2c18366bd530b85d0b6`.
- The results JSON contains each mode's selected checkpoint and prediction
  hashes. It currently labels the Torch version under a `python` metadata key;
  the reporting code has since been corrected. This metadata naming issue
  does not affect training or saved predictions.

Repository verification on this branch passed `ruff check .`, all CPU tests
(`2,783 passed, 11 skipped`), and the CI-equivalent model-manifest,
dataset-manifest, catalog, and dataset-manifest audit commands.

No model is promoted or productized. The present evidence favors improving
task/source coverage and cross-group validation before increasing recurrent
capacity. It does not establish the target's four-modality quality, does not
evaluate audio or image in this task, and does not meet the 95–98% quality
aspiration.

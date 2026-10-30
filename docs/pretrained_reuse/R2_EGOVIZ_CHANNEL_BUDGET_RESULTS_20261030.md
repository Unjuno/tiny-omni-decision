# EgoViz channel mixer under a matched 512-epoch budget

Both fixed CPU runs completed all 512 epochs / 5,632 optimizer updates. The
[protocol](R2_EGOVIZ_CHANNEL_BUDGET_PROTOCOL_20261030.md), runner and model code
were committed as `ffb99d85425bee53736c5e4989a8b78bbfc62cd2` before execution.
Each arm used the same 332 training windows and 307 development windows, seed,
initial projection weights, batch order and optimizer settings. The candidate
adds 3,072 trainable channel-mixing weights to the 81,920-parameter baseline.
Its zero-initialized gate matched baseline initial logits within `2.99e-8`.

| Arm / time | Train Accuracy | Development Accuracy | Development NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| Baseline, selected epoch 16 | 90.36% | 56.03% | 1.2147 | 0.6582 | 0.2260 |
| Channel mixer, selected epoch 16 | 91.87% | 57.00% | 1.2072 | 0.6539 | 0.2335 |
| Baseline, final epoch 512 | 100.00% | 41.37% | 1.3110 | 0.7100 | 0.0919 |
| Channel mixer, final epoch 512 | 100.00% | 37.46% | 1.3218 | 0.7161 | 0.0497 |

Both best checkpoints were chosen by minimum development NLL with `1e-6`
minimum improvement **among the 65 scheduled evaluations** (epoch 1 and every
eight epochs). The fixed final checkpoint was preserved separately. The model
does not become accurate merely because ECE decreases at epoch 512: its
accuracy and confidence both fall. The channel mixer modifies pooling and adds
parameters, so this experiment cannot attribute a difference solely to either.

Selected-checkpoint Accuracy changed by **+0.98 percentage points**, paired
20-session bootstrap 95% interval **[-1.98, +3.86]**. Baseline-minus-candidate
NLL was **+0.0075** [+0.0030, +0.0123]; Brier difference was +0.0043
[+0.0016, +0.0072]. These selected-checkpoint intervals do not account for
checkpoint selection. At fixed epoch 32, the Accuracy interval also included
zero; NLL difference was +0.0158 [+0.0093, +0.0224]. At epoch 512, the
Accuracy difference was **-3.91 points** [-10.00, +2.01], and the NLL interval
included zero. All are paired 2,000-resample percentile intervals with seed 17.
The observed small early NLL difference and uncertain Accuracy do not justify
promoting the mixer or claiming general improvement on unseen users.

## Longer learning curves

| Epoch | Baseline train Accuracy | Baseline dev Accuracy / NLL | Mixer train Accuracy | Mixer dev Accuracy / NLL |
|---:|---:|---:|---:|---:|
| 16 | 90.36% | 56.03% / 1.2147 | 91.87% | 57.00% / 1.2072 |
| 32 | 95.48% | 53.75% / 1.2258 | 96.39% | 55.05% / 1.2100 |
| 128 | 98.49% | 50.49% / 1.2610 | 99.70% | 48.53% / 1.2549 |
| 256 | 100.00% | 47.88% / 1.2854 | 100.00% | 44.30% / 1.2847 |
| 384 | 100.00% | 43.97% / 1.3054 | 100.00% | 42.35% / 1.3039 |
| 512 | 100.00% | 41.37% / 1.3110 | 100.00% | 37.46% / 1.3218 |

Rolling training CE fell from epoch 16 to 512 (baseline 0.9931 to 0.7304;
mixer 0.9768 to 0.6464), while development NLL worsened and Accuracy fell.
The table uses separately recomputed full-set train Accuracy, not rolling
training CE, for the train/development comparison. There is clear overfitting
within this fixed run. No late generalization recovery appeared through 5,632
updates; this does **not** prove that grokking, a different dataset, longer
training or another architecture cannot help. One seed and a previously used
development split limit generalization of the finding.

Each arm consumed **332 unique windows / 24 source sessions**, with **169,984
total window exposures** and 5,312 exposures through its selected epoch.
All 512 sample-order permutations matched; SHA-256
`264b2f81ad874d529a0bc37c4cb7eddfcb225ddc5221a33ca3ca3fc27f594da4`.
The original early-stopped run used a different order generator and evaluation
schedule; this is a new matched comparison, not a resume or identical replay.

## Integrity, resources and artifacts

The already-validated upstream aggregate was rechecked against SHA-256
`2655a6bdb654e7ab11b2c343286c2a2e007170412a5b554689843e6262381aa8`.
All file hashes matched; an additional in-flight audit checked 639 unique sample
IDs, train/development session, chunk and raw-media-hash separation, four
distinct candidates per row, and target-to-window indices. It does not prove
different participants or sites. No publisher action labels or sealed media
were read. Previous artifacts and source checkpoints were retained.

The running source and imported scorer, policy and baseline trainer matched the
frozen commit in an in-flight snapshot and again after completion. The runner
itself did not independently record the imported source hashes at process
start; this limits the provenance claim. A fresh process reloaded **all 14**
best/fixed checkpoints, checked the saved sample IDs, session IDs and targets
against the hashed corpus, and reproduced each saved logit exactly. Best-arm
option permutation checks passed. Fixed-epoch checkpoints' hashes were computed
during reload; best/final hashes also matched the original run report.

Four CPU threads on an i7-12700H were used. Baseline wall time was **47.13 s**
and channel mixer **171.81 s** (about 3.65 times longer); this includes training,
scheduled evaluation and checkpoint work. A separate three-step preflight
estimated 27.03 s and 130.26 s of update-only time and did not include those
overheads. It reported peak working set 1,075,261,440 bytes; full-run peak
memory was not sampled. No GPU or cloud compute was used. Best checkpoint sizes
were 329,212 and 341,814 bytes, excluding encoders; these are experimental
readouts and do not establish a complete four-modality runtime budget.

The complete curves, metrics, source/input evidence and paired comparisons are
in the [machine-readable result](../../manifests/pretrained_reuse/egoviz-channel-budget-results-20261030.json).
External artifacts are under
`C:\CodexArtifacts\egoviz-channel-budget-20261030-attempt-01`:

- `results.json`: SHA-256 `c2a8f6ddcc7c1bdf727ab4bc6b4c1533e9d34efc5c2ceabc114f125695dae8ae`.
- `reload-and-comparison.json`: SHA-256 `e0503252cd4d837da4186f00fdf09cbd9b77dc1322cb62e6a96c44bcddc9aae5`.
- `learning-curves.csv`: SHA-256 `a09c945e4987615f6da47fba9bca0a27be8dd7bdd1f5faf1b7ae65bded493897`.
- Best and fixed-epoch state files, predictions, status, and source/input audit
  receipts are retained. Large artifacts are not committed.

Exact original commands, from the repository root:

```powershell
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:HF_HUB_OFFLINE = '1'
$py = 'C:\Users\junny\AppData\Local\Programs\Python\Python311\python.exe'
$base = 'C:\CodexArtifacts\egoviz-unseen-group-acquisition-20261030-attempt-02'
$out = 'C:\CodexArtifacts\egoviz-channel-budget-20261030-attempt-01'
& $py -m scripts.train_egoviz_channel_budget --corpus "$base\full-aggregate" --baseline-report "$base\readout-results-cpu-01\readout-results.json" --output $out
& $py -m scripts.verify_egoviz_channel_budget --corpus "$base\full-aggregate" --run-dir $out --output "$out\reload-and-comparison.json"
```

Existing output paths reject overwrite. This is a narrow video-plus-sensor
timestamp task, not text/image/audio/video decision accuracy or a deployed
model. Next work should prioritize more independent source groups and a
condition-sensitive task, then evaluate any architecture change under matched
conditions before a sealed audit or compression.

# Native state binding: matched versus independent batches

## Why this bounded comparison

The completed common native baseline and LSTM fit do not establish useful temporal
reasoning. The fixed TRAIN contrast diagnostic finds almost no correct state-to-
answer changes in consumed text/video. Before expanding recurrence, test whether
batch correlation affects learning of the existing native readout. This is one
seed17 comparison, not a sweep, quality claim, or encoder-capacity ceiling test.

Use the same attention readout (889,345 trainable; 196,254,874 total inference
parameters before quantization), frozen encoder banks, CE loss, full candidates,
and initial state as the previous common baseline. No LSTM/xLSTM, new data,
teacher weights, quantization, audit access, or runtime promotion.

## Prospective fixed contract

- Exactly the original baseline's 8,192 sample **occurrences**: 6,656 unique
  questions, 1,536 repeats, 5,107 observations, 3,114 media assets, 3,382 groups.
  Each modality contributes 2,048 occurrences. Every source/task count is retained.
  These are not all 100,580 TRAIN questions.
- Both new arms use eight one-question microbatches/update: two per modality,
  summed CE divided by eight, 1,024 updates, no performance early stopping.
  Original four-microbatch/2,048-update fits are historical references, not the
  causal control. Update count and LR history differ from that older experiment.
- AdamW lr .001, betas (.9,.999), eps1e-8, weight decay .01, clip norm1,
  FP32, TF32 off, deterministic seed17. Warmup31 updates then cosine to zero at
  1,024; the zero-LR final call still consumes eight examples and computes gradients.
- Initial checkpoint is `native-common-baseline-fit-01/initial.pt`, never a
  trained best/final checkpoint. A separate TRAIN-only32-update smoke is discarded
  for each arm. Both full fits start fresh with identical state and RNG.
- `independent`: hash-order occurrences within modality into pairs.
- `matched`: group exact source/task/question/full candidate text set; greedily
  pair different observations and correct candidate texts. Remaining occurrences
  become deterministic fallback pairs, including original text and negative-only
  event cases. No filtering, resampling, target remapping, or duplicate generation.
- Hash-order pairs into updates. Candidate permutation is keyed by immutable
  original pool occurrence identity, so each occurrence sees exactly the same
  permutation in both arms, regardless of its update.
- This changes batch correlation **and sequence**. CE remains additive; there is
  no pairwise contrastive loss. It does not mathematically enforce state binding.
  A negative result only closes this batching intervention at this budget.
- Full validation24,346 (including all47 text slices), same IDs/order/options.
  Evaluate0,128,...1,024. Select hierarchical task/source/modality macro NLL,
  min_delta1e-5, earliest tie, with initial0 eligible. Keep best and final separately.
- Same fixed TRAIN probe1,456. It is mostly unconsumed; neither rolling train
  metrics nor that probe is a memorization gap. Report its overlap explicitly.
- Only after fitting: evaluate the already frozen1,250 TRAIN contrast pairs at
  the fixed endpoint. These constructed pairs are a binding diagnostic, not
  independent validation or a new blind benchmark. Do not select checkpoints
  from them. Replay full best/final validation and endpoint contrast arrays in
  a fresh process; require exact equality.
- Resource guard: two hours/process,6GiB CUDA allocated,10GiB process working set,
  minimum2GiB free disk. Stop for numerical/integrity/resource failures; never
  overwrite outputs. No resume claim: optimizer/RNG are saved but resume is not
  implemented by this runner.

## CPU feasibility before fitting

Plan folder: `C:/CodexArtifacts/pretrained-reuse-r2/native-matched-plan-01`.
All8,192 occurrence identities are retained exactly once in each new order;
original repeated questions remain separate occurrences. No task is removed.

| Modality | Independent eligible pairs | Matched eligible pairs | Pairs/arm |
|---|---:|---:|---:|
| Text | 3 | 545 | 1,024 |
| Image | 238 | 1,024 | 1,024 |
| Audio | 37 | 564 | 1,024 |
| Video | 1 | 372 | 1,024 |
| Total | 279 | 2,505 | 4,096 |

Matched fallback1,591 pairs remain; hash-greedy pairing does not claim a maximum
matching. Full pool counts and hashes are saved in the plan receipt. Both plan
files, permutations, corpus/query receipts, and initial checkpoint are pinned in
`configs/pretrained_reuse/native_matched_batch_fit_v1.json`. Runtime regenerates
and compares the plan, verifies every source/input hash before/after execution,
and saves the effective config, full order, all per-example scores and learning
curves. The old runner/config/artifacts remain unchanged.

## Interpretation fixed before results

Primary comparison is the two new fixed1,024 endpoints; separately show each
validation-selected checkpoint. Report every modality's Accuracy/NLL/Brier/ECE,
all task metrics, minimum task/modality accuracy, STARSS positive recall, and
consumed versus unconsumed contrast behavior. Changes in all-no Accuracy are not
positive event recognition. Report selected versus entire-run consumption.

Support for a useful batching intervention requires lower endpoint macro NLL,
increased both-correct rates on consumed text **and** video pairs, and no modality
Accuracy loss greater than2pp or NLL increase greater than .05. These are pilot
research interpretation rules, not product quality floors. Otherwise label mixed
or unsupported and retain both results. Report paired video scene uncertainty
from saved validation predictions (5,000 draws, seed17, percentile95% linear
quantiles); it is conditional on this observed development set and one seed.

Regardless of the outcome, <=200M and four-modality95–98%, conditional/joint/OOD
quality, calibration and deployment remain the product requirements. No new
architecture, larger budget, seed retry or prompt tuning is automatically enabled.

## Commands

Set the existing local dependency PYTHONPATH; do not upgrade packages. For each
policy (`independent`, `matched`) use separate `native-batch-<policy>-<phase>-01`
directories under `C:/CodexArtifacts/pretrained-reuse-r2`:

```powershell
python scripts/train_native_matched_batches.py --phase smoke --policy independent --output <new-smoke-dir>
python scripts/train_native_matched_batches.py --phase train --policy independent --output <new-fit-dir> --reference <smoke-dir> --reference-sha256 <verified-receipt-SHA256>
python scripts/train_native_matched_batches.py --phase replay --policy independent --output <new-replay-dir> --reference <fit-dir> --reference-sha256 <verified-receipt-SHA256>
```

Repeat for `matched` only after the independent process exits. Save actual
arguments, source commit, times, environment, VRAM, hashes and failures. Smokes
and planned commands alone do not count as completed fitting.

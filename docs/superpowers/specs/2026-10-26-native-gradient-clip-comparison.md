# Native Decision gradient-clipping comparison

## Purpose

Test whether the fixed global gradient-norm cap of 1.0 materially limits the
existing frozen-feature Path A readout. A TRAIN-only audit sampled 32 updates
from one selected checkpoint: 31 exceeded norm 1.0, with mean pre-clip norm
1.801 and mean applied factor 0.622. Pairwise modality gradient directions
were variable and near zero on average, so the evidence supports testing clip
scale, not changing modality weights or adding modality-specific capacity.

## Approaches considered

1. Keep norm 1.0 and make no intervention. This preserves the baseline but does
   not test the observed near-ubiquitous clipping.
2. Raise the norm to 2.0. This is the selected one-factor comparison: it reduces
   clipping pressure while preserving the model, optimizer, data, schedule and
   selection rule.
3. Disable clipping. This is a larger optimization change and could produce
   unstable updates; it is deferred until the bounded 2.0 comparison supports it.

## Fixed experiment

- Control: the existing seed-17, norm-1.0 `native-long-budget-fit-01` run.
- Candidate: a fresh seed-17 initialization with global norm cap 2.0.
- Preserve the original 32,768 updates, repeat pool, 8 microbatches per update,
  AdamW, LR 0.001, 62-update warmup, full-horizon cosine, weight decay 0.01,
  gradient accumulation, option shuffles, validation IDs, metrics, and
  macro-NLL checkpoint selector.
- Frozen encoders/features, 889,345 trainable readout parameters and the
  196,254,874-parameter pre-quantization composition remain unchanged.
- Best and final weights remain separate. No performance early stop, extra seed,
  audit access, quantization, encoder update or data change.
- Compare the candidate's fixed 512/2,048/4,096/8,192/32,768 observations and
  selected checkpoint with the preserved norm-1 run. Keep all per-modality and
  per-task Accuracy/NLL/Brier/ECE, calibration and retention failures visible.

The gradient audit used one selected checkpoint and a fixed TRAIN sample; it is
not validation evidence and does not establish that clipping caused prior
quality trade-offs. A result from this single-seed development comparison does
not qualify the product or establish a general optimizer rule.

## Runtime and safety

Use the local RTX 3080 Laptop only. Use the cooperative HotUGPU reservation
guard; the child must fail closed unless the guard is its sole verified GPU
owner and its declared memory snapshot still matches. Never stop or modify the
Orbitune process or its files. Do not start another GPU task while this run is
active. If the guard cannot obtain a cooperative yield within its bounded wait,
record the attempt as not started and leave the goal active.

All outputs use new `C:/CodexArtifacts/pretrained-reuse-r2/native-long-budget-clip2-*`
directories. The existing config, corpus, features, norm-1 checkpoint and
artifacts remain read-only. Run a separate smoke first; train only after it
passes the existing resource and integrity gates. No cloud compute or cost.

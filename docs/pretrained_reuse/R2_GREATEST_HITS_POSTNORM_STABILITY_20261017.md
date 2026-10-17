# Greatest Hits AV post-attention normalization candidate

## Scope and execution state

This is a numerical-stability investigation after the E0G0 AV-joint run stopped
on large finite losses and gradient norms. Its source was the pinned matched-run
commit `8a5440da334e74e970d80a7059ca5bb99a7248cb`. The E0G0 run stopped at 5,385
flushed optimizer updates; its selected checkpoint is step 4,096, SHA256
`7e73ca5a5cdd8efde8277e37876dd2b604aa840e0dafe6566cc3afa856f0e87f`. Existing
runs and checkpoints were read only.

This report records a CPU diagnostic and an opt-in code candidate. No optimizer
updates, GPU model loads, validation evaluations, sealed-audit reads, or cloud
compute occurred in this stage. At the time of the environment check, an
unrelated Orbitune process occupied about 15.3 GiB of the local GPU; therefore no
new GPU run was started. The result below is not a quality result and does not
retroactively repair E0G0.

## Exact saved-checkpoint probe

The probe used the four AV TRAIN examples presented at E0G0 step 4,149, with
their saved candidate orders and labels, and extracted only the already cached
features. It loaded checkpoint 4,096 on CPU. This is a diagnostic on consumed
TRAIN examples, not an independent estimate of generalization.

| TRAIN example | Pooled audio/video mean row norm | Candidate-attention output norm, raw A/V | Mean norm after shared output LayerNorm | Max absolute correction, raw → normalized | CE, raw → normalized |
|---|---:|---:|---:|---:|---:|
| `2015-03-30-01-08-49/0006/reaction` | 26.82 / 36.92 | 303.20 / 622.01 | 13.86 / 13.86 | 1761.18 → 4.04 | 2.5911 → 2.5827 |
| `2015-10-03-14-50-14-181/0022/action` | 20.71 / 38.68 | 257.52 / 578.38 | 13.86 / 13.86 | 1439.65 → 4.10 | 2.5685 → 2.5664 |
| `2015-03-29-17-50-50/0029/material` | 19.17 / 42.03 | 258.15 / 568.46 | 13.86 / 13.86 | 1423.31 → 4.18 | 2.1748 → 2.1746 |
| `2015-03-15-19-34-38/0037/reaction` | 23.16 / 33.67 | 282.60 / 607.66 | 13.86 / 13.86 | 1609.11 → 4.05 | 0.0486 → 0.0489 |

The saved residual therefore has a concrete scale problem between query-conditioned
candidate attention and the correction MLP. Applying the post-attention norm to
these already-trained weights reduces the correction magnitude by roughly 350–400x
while leaving these four CE values and predictions nearly unchanged. This is
evidence that scale can be bounded at that interface; it is not evidence that
training gradients become stable, that validation improves, or that the norm is
the sole cause of the earlier spikes.

## Isolated code candidate

The matched runner now has an explicit `--candidate-output-norm` ablation. It
applies one shared trainable LayerNorm after query-conditioned attention to both
audio and video candidate features, before the conditional gates and correction
scorer. The default remains off. With it off, the model retains the legacy
checkpoint schema and tensor layout. With it on, checkpoint metadata records a
distinct schema/policy and cross-policy loads fail closed.

The candidate adds 384 trainable parameters:

| Ledger | Parameters |
|---|---:|
| Existing AV readout | 1,732,420 |
| Candidate shared LayerNorm | 384 |
| Candidate AV readout | 1,732,804 |
| Frozen encoder reference | 195,365,529 |
| Candidate reference total before quantization | **197,098,333** |
| Remaining under 200M | **2,901,667** |

The optional runner path copies the exact legacy same-arm initial tensors and
adds only the identity-initialized norm state; it retains fresh optimizer state.
The effective config records this policy, so a candidate smoke/training reference
must use the same setting. No such smoke or training run has been performed yet.

The separate gate head remains small at 61,634 parameters and 61,568 MACs per
option. That arithmetic supports treating its parameter/compute cost as small
relative to this readout, but no end-to-end latency or quantized package size was
measured here.

## Verification and next action

Targeted CPU checks pass: 50 tests across native AV interaction, matched runner,
and matched runtime. They cover legacy-default state compatibility, policy-bound
checkpoint loading, candidate parameter accounting, and finite gradients through
the new norm. Ruff and `git diff --check` are run after formatting.

Next, wait for the local GPU to become available through its configured yield
behavior, then run a separate candidate smoke in a fresh output directory. The
smoke must verify actual end-to-end numerical stability, peak VRAM, and measured
runtime before any long fit is considered. If stable, compare the norm candidate
against the same E0G0 protocol on the unchanged data/order and report the safety
stop as the prior incomplete run. Preserve the existing E0G0 artifacts and all
other matched arms; do not claim a capacity limit, model quality gain, R2
completion, or product readiness from this ablation.

# Modality-preserving gated temporal readout result (2026-10-26)

## Question and scope

The previous query-conditioned temporal candidate learned a gate that assigned
almost all mass to audio. This follow-up kept independent audio and video
projections visible to the recurrent layer alongside the gated mixture. It
changed only this readout architecture: data, frozen encoders/features, seed,
optimizer, learning rate, batch, epoch count, validation schedule, and selector
were held fixed.

This is a synthetic task diagnostic, not natural AV transfer or product
quality evidence. The same development validation split was used for
checkpoint selection; it is not blind. No sealed audit was opened, and no old
data, weights, or reports were changed.

## Model and protocol

- Two-layer, width-112 LSTM receives four per-step vectors: projected audio,
  projected video, their query-conditioned gated mixture, and projected
  query. Both individual modalities therefore have a direct path into the
  recurrent layer even when the auxiliary gate collapses.
- Trainable parameters: 521,476, 6.11% below the reference temporal LSTM's
  555,458 and within the 10% parameter-match envelope.
- Seed 17, AdamW, LR 0.001, weight decay 0.0001, batch 32, 128 epochs,
  validation every four epochs. Selector: minimum validation NLL, strict
  improvement only, earliest tie retained. Best and terminal weights are
  separate.
- Same 4,096 train / 1,024 validation rows and exact same validation IDs as the
  original synthetic AV run. The sampler permutation SHA-256 also matches the
  reference: `7988a28d484c35ca071cc07a57e9f8c509860e28c175aa7e49d097ee0afe60e9`.
- Completed 16,384 optimizer updates and 524,288 example presentations. The
  selected step 3,584 represents 114,688 presentations; all 4,096 unique train
  rows had already appeared. This is repeated-epoch synthetic training, not
  524,288 distinct examples.

## Learning curve and same-validation comparison

The candidate stayed at chance on train and validation through the full
budget. The original concatenated-input temporal LSTM solved the synthetic
rule under the same seed and run settings.

| Step / selected checkpoint | Validation Accuracy | NLL | Brier | ECE | Online train CE | Online train Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| 512 | 0.5000 | 0.693162 | 0.500014 | 0.002689 | 0.693481 | 0.4944 |
| 2,048 | 0.5000 | 0.693155 | 0.500008 | 0.002030 | 0.693268 | 0.4858 |
| 3,584 (best) | 0.5000 | 0.693147 | 0.500000 | 0.000106 | 0.693301 | 0.4976 |
| 4,096 | 0.5000 | 0.693148 | 0.500001 | 0.000667 | 0.693281 | 0.4858 |
| 4,608 | 0.5000 | 0.693147 | 0.500000 | 0.000051 | 0.693353 | 0.4907 |
| 8,192 | 0.5000 | 0.693147 | 0.500000 | 0.000409 | 0.693200 | 0.4863 |
| 11,264 | 0.5000 | 0.693148 | 0.500001 | 0.000522 | 0.693272 | 0.4985 |
| 15,360 | 0.5000 | 0.693156 | 0.500009 | 0.002096 | 0.693213 | 0.4922 |
| 16,384 (terminal) | 0.5000 | 0.693148 | 0.500000 | 0.000491 | 0.693207 | 0.5000 |

| Readout / best checkpoint | Trainable parameters | Step | Accuracy | NLL | Brier | ECE | Mean confidence |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original temporal LSTM | 555,458 | 11,264 | 1.0000 | 0.000585 | 0.000278 | 0.000493 | 0.999507 |
| Bidirectional conditional AV | 513,412 | 3,584 | 0.5000 | 0.693147 | 0.500000 | 0.000022 | 0.500022 |
| Query-conditioned temporal AV | 528,388 | 3,584 | 0.5000 | 0.693147 | 0.500000 | 0.000087 | 0.500087 |
| Bypass-conditioned temporal AV (this run) | 521,476 | 3,584 | 0.5000 | 0.693147 | 0.500000 | 0.000106 | 0.500106 |

The four best checkpoints were evaluated by one CPU evaluator invocation using
the same 1,024 validation IDs, cached features, controls, and metric code.
Bootstrap differences from the LSTM reference are descriptive over 128 groups
(10,000 paired draws, seed 17), on the validation cohort used to select
checkpoints; they are not blind-test intervals. Candidate-minus-LSTM Accuracy
was `-0.5000` with percentile interval `[-0.5000, -0.5000]`.

## Post-selection diagnostic

The selected gate assigned mean weights `[0.002786, 0.997214]` to
`[audio, video]` over train time steps and `[0.002786, 0.997214]` over
validation time steps. This reverses the previous candidate's audio collapse:
the current gate almost entirely selected video. On the same first 32 seed-17
training rows, one read-only backward pass produced projection gradient norms
of `8.58e-6` (audio), `2.55e-5` (video), and `6.90e-8` (gate); temporal LSTM
gradient norm was `2.85e-3`. This confirms the bypass retained a video
gradient, but gate learning was nearly inactive and the full classifier still
did not learn the rule. One minibatch diagnostic cannot establish a general
optimization cause.

## Compute and artifacts

- Hardware/software: NVIDIA GeForce RTX 3080 Laptop 16 GiB; Python 3.11.9;
  PyTorch 2.6.0+cu124; CUDA 12.4.
- Training used the cooperative HotUGPU reservation: 2,418 MiB reserved;
  no other process was stopped. Runtime 123.24 s; peak allocated/reserved
  memory 456,042,496 / 473,956,352 bytes. CPU saved-checkpoint evaluation and
  controls took 43.38 s. Cloud cost `$0`.
- CPU smoke completed 2,048 smoke updates and exact best reload
  (`max_abs_logit_error=0`). It is separate from the full run.
- Source commit `4206ee1e387cd681b6dd8202fd10c69d5ad2f995`; evaluation CLI
  multi-reference support commit `c0b51f8`. Config SHA-256:
  `b5f46b177b56ee155baa5c78a060339402acb503776bfc9296d22ea45ae529ee`.
- Training receipt SHA-256:
  `6091b7e12b87a568ee244c8f1ef922af91d3f07f6089c9acbdbf33e9dde21d63`;
  history SHA-256:
  `52ac13bd216123bb64f4d7c64cb75e4d82f67be0ba6ca1a5d75ed230efc44c9d`;
  best weights:
  `b8644aef4db864de67749c9ff14a2abec4f021479e0936b7ae7df1c2b764d2ee`;
  terminal weights:
  `ac03cb0ff89f8e2d01a91678738b2955119bb3f36f2d57d2c7c6041af5a63f79`.
- Evaluation receipt SHA-256:
  `5c25a02a75ef395637d34dd24e70b522ceeae31714c63d92c3fa659abdd22fcb`;
  predictions SHA-256:
  `fd4af1ef5c481134367dac89236da6d448a0ba89281cdae8eaa0676374b4039a`.
- External paths:
  `C:\\CodexArtifacts\\pretrained-reuse-r2\\synthetic-av-bypass-conditioned-temporal-v3-attempt-01`
  and
  `C:\\CodexArtifacts\\pretrained-reuse-r2\\synthetic-av-bypass-conditioned-temporal-v3-eval-attempt-01`.
  No media or weights were added to Git.

## Interpretation and next experiment

This bypass-gated architecture is **not supported** on the current synthetic
task at this width, initialization, and budget. The two learned gated
architectures collapsed to opposite modalities but both remained at chance;
the unprojected, direct-concatenation LSTM solved the task. This leaves
projection/readout optimization as a plausible issue, while not proving it.

A narrowly scoped next experiment can preserve the exact baseline LSTM input
and initialization, adding only an identity-initialized residual scalar gate
that rescales but cannot zero either raw modality. Its extra parameters fit
within the existing 10% envelope. Hold all other conditions fixed. Do not
interpret another synthetic win as natural AV quality evidence.

# CLEVRER temporal pooling control (predeclared)

## Question

Test one narrow hypothesis: the current VideoPrism readout may lose useful
time-position information when it averages all spatiotemporal tokens into one
768-dimensional vector. Keep VideoPrism, MiniLM, the existing CandidateScorer,
frame count, source data, and labels fixed. Train only eight shared scalar
weights over frame-position-pooled VideoPrism features.

This is a development diagnostic. The readout was selected on an earlier
validation split; this probe's training split is the same data used to fit the
readout, and its evaluation scenes were evaluated once in the preceding
frame-reversal control. Results cannot promote a product model or establish a
general video capability.

## Source audit and frozen data

- VideoPrism `FactorizedEncoder` applies a spatial encoder per frame, adds
  temporal positional embeddings, then applies its temporal encoder. Its
  output is `[batch, time * spatial_tokens, 768]`.
- The current extractor averages that complete token axis. This removes the
  explicit time-position axis from the readout input, though temporal context
  may already be encoded in each token.
- Source dataset: `MIT-IBM/CLEVRER@98b842082ba4f7c18b6b9e3f39145871782a65ef`.
  Training uses the frozen 256-scene, 512-question train split. Evaluation
  uses all 400 questions from the separate 200-scene descriptive holdout
  (one static and one temporal descriptive question per scene). Preserve
  source question text, options, and targets.
- Enforce zero scene and media-hash overlap between train and evaluation.
  Report repeated normalized question/options content; do not filter rows.
- VideoPrism checkpoint SHA-256:
  `61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29`.
  Eight uniform endpoint-inclusive frames at 288 px; CPU JAX extraction.
- MiniLM revision `4ca70771034acceecb2e72475f72050fcdde4ddc`, weights SHA-256
  `cf1e4e2d420c664973037c3c73125d7a8fc69952495093ef8f50596f8943a433`.
- Fixed VideoPrism+MiniLM CandidateScorer SHA-256
  `f68b04a4079c537772938d27c6456841c5eed19595ab49d0f42dd014211f0dbc`.

## One-factor procedure

1. Extract `[8, 768]` features by averaging spatial tokens separately at each
   final temporal position. If float32 two-stage reduction misses the existing
   pooled feature by roundoff, add the same per-channel offset to all eight
   positions for that asset. This preserves temporal position differences and
   uses no labels. Require the uniform mean to reproduce the existing pooled
   feature cache within maximum absolute error `1e-5` on every train and
   evaluation asset.
2. Baseline: uniform weights `[1/8] * 8`, passed through the untouched fixed
   scorer.
3. Candidate: initialize eight trainable logits at zero; softmax them into
   shared nonnegative temporal weights. Optimize only those eight logits with
   Adam (`lr=0.05`), full-batch CE on the 512 training questions, exactly 200
   updates, seed 17. Freeze the two encoders and every scorer parameter. Do
   not monitor or select on evaluation metrics; evaluate the final 200-update
   weights once.
4. Report train and evaluation Accuracy, NLL, Brier, ECE-15; temporal and
   static descriptive subsets; macro by question type; learned weights;
   paired 10,000-replicate scene bootstrap (seed 17); and exact prediction
   rows. Preserve all examples and disclose content repetition.

The uniform feature mean and baseline scores must reproduce the existing
forward caches and saved predictions within declared numeric tolerances. A
failure stops evaluation rather than weakening a hash or parity gate. This
probe adds only eight scalar parameters and does not alter the saved readout or
any prior feature cache.

## Interpretation

This tests a restricted position-weighted pooling rule. A gain would support
testing richer temporal readouts; no gain would not show that temporal
positions or VideoPrism's temporal representation are useless. Either result
is only on this small, reused development corpus and fixed scorer.

## Measured result (2026-10-11)

The frozen framewise features were extracted for 456 unique videos on CPU JAX
(`0.6.2`), taking 586.16 seconds with a 4.16 GB peak working set. A first
read-only parity preflight found a maximum absolute float32 reduction delta of
`3.62e-5`; evaluation stopped before optimization. A separate cache was then
centered with the same label-free per-channel offset at all eight positions,
preserving frame-to-frame differences. The resulting uniform-mean parity was
`2.86e-6`, below the predeclared `1e-5` bound. Existing source feature caches
were not modified. Recomputed baseline probabilities matched the saved forward
predictions within `5.96e-8`.

The local RTX 3080 Laptop GPU run started at `2026-10-11T08:28:36+09:00` and
finished at `2026-10-11T08:32:17+09:00`. It trained eight scalar parameters
for exactly 200 Adam updates on 512 questions from 256 scenes. Training NLL
changed from `1.178122` to `1.177720`; train accuracy changed from `0.4395` to
`0.4355`. The 400-row evaluation used 200 held-out scenes, with 200 temporal
and 200 static questions. It is previously observed development validation,
not a blind evaluation. Exact normalized question/options content overlaps
between the splits: 102 unique keys and 241 of 400 evaluation rows. Scene and
media identities remain disjoint.

| Evaluation subset | Pooling | Accuracy | NLL | Brier | ECE-15 |
|---|---|---:|---:|---:|---:|
| Temporal descriptive (200) | Uniform | 0.4350 | 1.214960 | 0.657060 | 0.08166 |
| Temporal descriptive (200) | Learned | 0.4400 | 1.214457 | 0.657126 | 0.09459 |
| Static descriptive (200) | Uniform | 0.3200 | 1.112112 | 0.638597 | 0.10284 |
| Static descriptive (200) | Learned | 0.3200 | 1.112205 | 0.638983 | 0.10799 |
| Macro by question type | Uniform | 0.3775 | 1.163536 | 0.647829 | 0.09225 |
| Macro by question type | Learned | 0.3800 | 1.163331 | 0.648054 | 0.10129 |

Learned minus uniform macro deltas were `+0.0025` Accuracy, `-0.000205`
NLL, `+0.000226` Brier and `+0.00904` ECE. The temporal subset gained one
correct answer net; static accuracy was unchanged. Paired 10,000-replicate
scene-bootstrap 95% percentile intervals included zero for all macro metrics:
Accuracy `[-0.005, 0.010]`, NLL `[-0.000719, 0.000303]`, Brier
`[-0.000108, 0.000554]`, and ECE `[-0.00967, 0.01783]`. The learned weights
were concentrated on positions 2 and 6 (`0.6210` and `0.3514`), but this
concentration did not produce a clear held-out gain.

The run took 221.42 seconds end-to-end, including 20.55 seconds for pooling
optimization and 1.36 seconds for text encoding; paired bootstrap accounted
for most remaining time. Peak CUDA allocation was 173,938,688 bytes. No cloud
compute was used. The saved report SHA-256 is
`e236330b83a39f6d460700d69ce6ef28b99f25f62bb2cd7d085b0c1b7dcee1b2`; saved
predictions SHA-256 is
`4604b582d0a6929bea1e7a95828a14a7c64c0eeebaa10b06a5831a01da450216`.
Artifacts remain under
`C:\CodexArtifacts\pretrained-reuse-r2\clevrer-temporal-pooling-control-v0`.

The evaluator report was written before code hashes had been added to its
serialization. The executed evaluator SHA-256 recorded at run time was
`c6616aa424df90455c94dab0045c99ab1f9df8d44ef0e77a57c0bad3114618cc`; the
framewise extractor SHA-256 was
`58a7afa21f4464e8efe12b4388f417c9cb33b39f8fff4a512a144e16984e9757`, and the
centering utility SHA-256 was
`970323381d4d3195095e52903a8829b17f9e99e1ff6d4a77b3265123e757f32b`. The
committed source contains the parity guard and the tested implementation; its
current SHA-256 values are tracked by Git after this change. Fixed scorer
trainer SHA-256 was
`89943d01ba780671841d6b79ea85fd7c0078ca98ef95226207c1242ae33acaa0`, and
candidate-scorer source SHA-256 was
`d8216db3030e81c3a598671c26536d4656d866979a57ef1429de7f2d280c78d4`.

The committed replay implementation and regression tests are also pinned here:

| File | SHA-256 |
|---|---|
| `scripts/evaluate_videoprism_temporal_pooling_control.py` | `e094bd39db9bb1eff9a8ad58339d1541054db5e8aa1aabe41cfbfd94db8d03cd` |
| `scripts/extract_videoprism_framewise_features.py` | `d2c52624ebfcbdfacad51f4b2f2600c243e94b5027effff3f07917b0082a5442` |
| `scripts/center_videoprism_framewise_cache.py` | `970323381d4d3195095e52903a8829b17f9e99e1ff6d4a77b3265123e757f32b` |
| `tests/test_videoprism_framewise_features.py` | `eb76d9c346203c2c19665ee76563888b0af0737cb571a26341301dabe0aabbc2` |

**Decision:** this small scalar pooling rule does not show a clear improvement
under the paired uncertainty and calibration guardrails. Do not proceed to
more pooling-weight tuning on these validation results. It does not test
richer temporal readouts, and cannot establish a limit of VideoPrism or of
frozen representations generally. The broader Path A joint-input and runtime
gates remain open.

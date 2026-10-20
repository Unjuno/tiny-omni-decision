# HoloAssist text-free temporal readout results (2026-10-20)

## Scope and status

This report closes the 2026-10-20 continuation documented as in progress in
`R2_HOLOASSIST_RECOVERY_20261020.md`. It records a text-free, frozen-feature
comparison on previously unused official training sessions. The decision
condition is the typed task category ID; no text embeddings or text values are
passed to the readout. This is a narrow HoloAssist action-quality task, not a
general-purpose multimodal benchmark or product-release result.

Execution status: **PASS for the three fixed-budget readout runs and the
predeclared validation interventions**. The separate paired bootstrap suite
completed all seven comparisons. No official validation/test or sealed-audit
payload was read. The feature, corpus, and checkpoint artifacts remain outside
Git and were not overwritten. No model was promoted.

## Frozen data and representations

The data selection is the previously frozen set of 1,042 train and 264
development-validation HoloAssist sessions (1,306 unique media assets), seed
20261019. Thirty-second causal windows contain 9,763 train rows and 2,498
validation rows. Of these, 9,351 train and 2,373 validation rows have labels;
unlabeled windows remain available as recurrent context and are excluded from
loss and metrics. The validation labels contain 719 positive and 1,654 negative
windows (30.30% positive). Train and validation share no sessions or media.
All source, annotation, media, event, and window hashes are recorded in the
recovery report.

Pinned frozen feature banks:

| Modality | Encoder and revision | Parameters | Feature shape | SHA-256 |
|---|---|---:|---:|---|
| Video | `google/videoprism-base-f16r288`, `be719a406d563b66f0ac969e7c94bab8e997c81a` | 114,365,184 | 12,261 × 768 float32 | `36c2a44417108484c4c6358aed485e3407601e79ddeed6e549dc6a151999c1b9` |
| Audio | `openai/whisper-tiny`, `169d4a4341b33bc18d8881c4b69c2e104e1cc0af` | 8,208,384 | 12,261 × 384 float32 | `39a667d1d23b01b7b102eabef0c5c21e0f19e422942fe341a4086be5780ce338` |

The banks use the same ordered 12,261 train-plus-validation window IDs. Text
features are excluded from the readout. The packed cache receipt is
`C:/CodexArtifacts/pretrained-reuse-r2/holoassist-textfree-unused-train-recovery-02/packed-cache-v3/receipt.json`
(SHA-256
`c3ba71bf118375ea3018e2cc076edbfe211f9f6a04318ea6b0baac89660484fc`). Packed
train and validation feature hashes are
`1c080f3dd3a661ca8e3134754dcc9ef0f88a9ac8df53f9b2fdfed5561aafd2d3` and
`2e760980460385cc615425f41e80d043fd66bc5e2d05c7ccf7d5742ee973b567`.

## Experiment

Source code was commit `39b1a7057c4ec0afaa04b3dbc09e1989d2fa654b`. The exact
effective run configuration is in the external `readout-v1-attempt-03`
`run-metadata.json` (SHA-256
`e20749b58a91b208340ebb3fd1913735da47623b57a80678df8c312f51a3e868`); config
SHA-256 `c023a3d4a8a86ddcb3964503241691717af3bd6aa6b9fb2165110596e7ab504f`.
Seed 17, 128-unit causal LSTM/readout, AdamW learning rate 0.001, weight decay
0.001, batch of 8 sessions, unweighted binary cross entropy, and no early
stopping were used. The three arms were video-only, audio-only, and joint
audio-video. All three had 332,932 trainable parameters. Each ran exactly 2,048
optimizer updates over 9,763 unique training windows, with 143,023 repeated
window occurrences (152,786 total training-window occurrences). Best
checkpoints were selected by lowest validation NLL, ties going to the earliest
step. Every arm completed the full fixed budget; selection did not stop the
run.

The local GPU readout run used an RTX 3080 Laptop GPU, Python 3.11.9, PyTorch
2.6.0+cu124. The run metadata records start 2026-10-20 13:45:03 JST and finish
13:47:08 JST. Peak allocated CUDA memory was 86,098,432 bytes (82.1 MiB) per
arm. All work used local hardware; cloud compute and paid compute cost: none.
The VideoPrism feature extraction ran on its pinned CPU/JAX path, and Whisper
feature extraction used the local GPU under the cooperative HotGPU guard. The
readout and extraction did not stop or modify other jobs. The three readout
runs completed while holding a 3,200 MiB HotGPU reservation.

## Validation results

All metrics below use the same 2,373 labeled development-validation windows.
Accuracy is prevalence-weighted; balanced accuracy is shown because the task is
imbalanced. ECE uses 15 bins. Best checkpoints were reloaded and replayed.

| Arm | Updates | Best step | Accuracy | Balanced accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Video-only | 2,048 | 1,024 | 0.7105 | 0.5702 | 0.5771 | 0.3919 | 0.0198 |
| Audio-only | 2,048 | 960 | 0.7509 | 0.6594 | 0.5405 | 0.3568 | 0.0369 |
| Joint audio-video | 2,048 | 960 | 0.7493 | 0.6652 | 0.5322 | 0.3507 | 0.0295 |

The best joint model improves over Video-only by 3.88 percentage points in
Accuracy, 9.50 points in balanced accuracy, and 0.0449 NLL. It does not improve
raw Accuracy over Audio-only: the difference is -0.17 points. Joint improves
NLL by 0.0083 and Brier by 0.0061 over Audio-only, with only small/uncertain
changes to other metrics. This supports a joint-vs-video benefit on this task,
but does not establish a material raw-Accuracy gain from adding video to audio.

### Learning-curve checkpoints

| Arm | Step | Rolling train CE | Rolling train accuracy | Validation accuracy | Validation balanced accuracy | Validation NLL | Validation Brier | Validation ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Video-only | 512 | 0.5589 | 0.7198 | 0.7029 | 0.5392 | 0.6063 | 0.4123 | 0.0498 |
| Video-only | 1,024 | 0.5480 | 0.7261 | 0.7105 | 0.5702 | 0.5771 | 0.3919 | 0.0198 |
| Video-only | 1,536 | 0.5213 | 0.7495 | 0.7008 | 0.5711 | 0.5929 | 0.4036 | 0.0386 |
| Video-only | 1,920 | 0.5187 | 0.7485 | 0.6991 | 0.5518 | 0.5983 | 0.4052 | 0.0602 |
| Video-only | 2,048 | 0.4960 | 0.7577 | 0.7075 | 0.5473 | 0.6125 | 0.4096 | 0.0820 |
| Audio-only | 512 | 0.5135 | 0.7514 | 0.7328 | 0.5890 | 0.5638 | 0.3770 | 0.0425 |
| Audio-only | 1,024 | 0.4927 | 0.7675 | 0.7459 | 0.6577 | 0.5432 | 0.3590 | 0.0417 |
| Audio-only | 1,536 | 0.4680 | 0.7797 | 0.7349 | 0.6019 | 0.5646 | 0.3746 | 0.0423 |
| Audio-only | 1,920 | 0.4566 | 0.7834 | 0.7349 | 0.6290 | 0.5728 | 0.3749 | 0.0588 |
| Audio-only | 2,048 | 0.4427 | 0.7930 | 0.7421 | 0.6180 | 0.5708 | 0.3727 | 0.0739 |
| Joint | 512 | 0.5075 | 0.7587 | 0.7387 | 0.6082 | 0.5530 | 0.3684 | 0.0334 |
| Joint | 1,024 | 0.4773 | 0.7807 | 0.7476 | 0.6581 | 0.5342 | 0.3529 | 0.0176 |
| Joint | 1,536 | 0.4461 | 0.7988 | 0.7484 | 0.6414 | 0.5567 | 0.3619 | 0.0454 |
| Joint | 1,920 | 0.4213 | 0.8022 | 0.7248 | 0.6552 | 0.5774 | 0.3768 | 0.0564 |
| Joint | 2,048 | 0.4045 | 0.8156 | 0.7505 | 0.6261 | 0.5909 | 0.3741 | 0.0897 |

Rolling train metrics aggregate recent sampled batches and are not a fixed-set
generalization measure. For all three arms, train CE continues falling while
validation NLL/calibration worsen after the validation-selected best point.
This is evidence of overfitting in later updates. The fixed budget still
completed as planned; the best checkpoint, rather than the endpoint, is used in
the comparisons.

## Fixed validation interventions and uncertainty

The joint best checkpoint (step 960) was evaluated with unchanged weights under
baseline, audio missing, video missing, audio time-reversed within each
session, video time-reversed within each session, and audio reassigned between
validation sessions matched by task type and window count. Session-cluster
paired bootstrap used 10,000 replicates, seed 1701, resampling 264 sessions
with replacement and retaining all 2,373 labeled windows. Intervals are
percentile 95% intervals. The paired validation prediction-ID hash is
`aa8e4a07f4e65229407481bd28bfeb404f5d71cc95141f9d736ef0df66b2c066`.

The table reports candidate minus baseline, so lower NLL is a negative value.

| Candidate relative to joint baseline | Δ Accuracy (95% interval) | Δ NLL (95% interval) | Δ Brier (95% interval) | Interpretation |
|---|---:|---:|---:|---|
| Video-only best | -0.0385 [-0.0585, -0.0188] | +0.0449 [+0.0225, +0.0670] | +0.0412 [+0.0232, +0.0590] | Joint is clearly better than video-only. |
| Audio-only best | +0.0017 [-0.0069, +0.0103] | +0.0083 [+0.0009, +0.0162] | +0.0061 [-0.0000, +0.0124] | No clear raw-Accuracy difference; audio-only NLL is slightly worse. |
| Audio missing | -0.1753 [-0.2199, -0.1329] | +0.1645 [+0.1210, +0.2087] | +0.1499 [+0.1123, +0.1890] | Strong evidence the joint model relies on audio. |
| Video missing | -0.0144 [-0.0259, -0.0032] | +0.0229 [+0.0134, +0.0330] | +0.0203 [+0.0125, +0.0286] | Video contributes a small, detectable benefit relative to no video on these IDs. |
| Audio time-reversed | -0.1209 [-0.1486, -0.0938] | +0.2229 [+0.1763, +0.2723] | +0.1687 [+0.1346, +0.2046] | Audio temporal alignment materially matters. |
| Video time-reversed | +0.0017 [-0.0063, +0.0101] | +0.0117 [+0.0048, +0.0204] | +0.0079 [+0.0025, +0.0147] | Little raw-Accuracy change; probabilistic scores worsen modestly. |
| Audio reassigned across sessions | -0.0497 [-0.0670, -0.0331] | +0.0750 [+0.0544, +0.0966] | +0.0591 [+0.0425, +0.0767] | Correct audio-to-session alignment matters. |

Interpretation is limited to the same development-validation sessions used for
checkpoint selection. These intervals are diagnostic and selection-biased; they
are not independent confirmation or a new blind evaluation. The evaluator's
input paths were the train cache, validation cache, feature receipt, and
readout output directory; its saved predictions match the fixed validation
IDs. The intervention `result.json` does not itself contain a
`sealed_audit_loaded: false` field. The evaluator now emits it in new result
files and a regression test checks it; the historical attempt-03 JSON remains
immutable. For that run, the absence of audit loading is supported by the
executable input path and prediction-ID check, not by that missing metadata
key.

## Decision and remaining limits

For this text-free HoloAssist action-quality task, audio dominates the learned
gate (mean audio weight about 0.775 across joint validation examples). The
joint model is materially better than video-only, and video missingness worsens
joint validation metrics, but joint raw Accuracy is essentially tied with the
audio-only arm. Video reversal leaves Accuracy nearly unchanged while modestly
worsening NLL/Brier. This is evidence of some video utility and weak temporal
order sensitivity in this particular setup, not proof that video has no
temporal information or that the fusion architecture is at its limit.

The result does not close R2's modality-native, image, sensor, cross-task, or
multi-input-required gates. It does not establish 95–98% quality, a
100–200M-parameter product, a <=200 MB package, post-quantization quality,
deployment latency, or general Omni Decision capability. Existing v1/v2
Teacher or QAT artifacts were not changed. No sealed audit, official
validation/test split, ternary quantization, Recovery LoRA, model promotion,
cloud run, or extra training seed was used.

### Reproduction artifact locations

All large data and model outputs are external to Git under
`C:/CodexArtifacts/pretrained-reuse-r2/holoassist-textfree-unused-train-recovery-02/`:

- `video-features-v1-attempt-04/` and `audio-features-v1-attempt-02/`: frozen media feature banks.
- `packed-cache-v3/`: train/validation caches and feature receipt.
- `readout-v1-attempt-03/`: effective run metadata, per-arm histories, best and final checkpoints, and validation predictions at every 64-step evaluation.
- `readout-v1-intervention-attempt-03/`: intervention predictions and metrics.
- `readout-v1-intervention-bootstrap-attempt-03/`: seven paired session-bootstrap reports and comparison index.

The selected joint checkpoint is step 960; audio-only is step 960; video-only
is step 1,024. Final step-2,048 checkpoints remain separately preserved.

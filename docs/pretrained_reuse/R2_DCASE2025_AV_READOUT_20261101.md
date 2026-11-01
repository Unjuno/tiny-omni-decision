# DCASE2025 frozen AV readout: one-shot dev-test result

## Status and scope

The frozen-feature audio/video candidate was evaluated once on the public
room-held-out DCASE2025 dev-test split. This is a development benchmark result,
not a sealed audit and not a product-quality claim. The evaluation does not
close the roadmap's independent multimodal-quality or both-input-required task
gates. No model was trained, recalibrated, selected, or changed using dev-test.

The candidate freeze was recorded before dev-test annotation content was
opened. It freezes the three 30-epoch readouts from dev-train, their config and
hashes. The checkpoint rule was the fixed final epoch, with no validation
selection. The readouts fit all 15,485 dev-train rows and have very high
training accuracy (audio 98.92%, video 94.70%, joint 99.62%); this signals
substantial overfit risk.

## Frozen evaluation inputs

- Source commit: `666dc5a772322399b0b1e979e407d8d284d31166`.
- Candidate freeze: `C:\CodexArtifacts\pretrained-reuse-r2\dcase2025-av-candidate-freeze-20261031-attempt-01\freeze.json`, SHA-256 `8dee609bfdc266af0016ccab70167df9566f677e9bcc7ef5db669aaeb640f675`.
- Frozen training result SHA-256 `a5399b5b099f13a205bba42ca1ff148a19f9e73b5367c433f9624a229bf4682d`; effective config SHA-256 `dc3c0d266b0542f927f00affc3af5446794cb0617da9e110fd4fdb8c0f073ce9`.
- Frozen checkpoint SHA-256: audio `1b8080899ded11eb8e0a6698b313036858622ed36139481aa6f877882cfec1d8`, video `01eed351dfd1a3a7271b65388ad0fe54689f75704a09cf5c68249b9e88f26b12`, joint `358e7185312b7de9856d77cb6d9e7b0b342fbf382e95a38f3db434d9ae62bcbc`.
- Dev-test preflight: PASS, 13,308 rows, 78 parent recordings, 7 rooms, 13 event classes; selected-ID order SHA-256 `c7e803da1692944b211a613255fcca23924bb8219ef5d6e94aac229cf0ef4e4d`.
- Frozen feature extraction: PASS, 13,308/13,308 rows valid. Audio feature SHA-256 `a663635d3ab1c1a105612da8e2d573ff602597c57089ac74c4cd5237ece2ebc2`; video feature SHA-256 `768f936e4948eb913ff6639b1fe913f6426badf0ac42ce3c4f17de5bdeb7d536`.
- Targets: 9,890 offscreen / 3,418 onscreen. No sealed-audit data was accessed.
- Evaluation output: `C:\CodexArtifacts\pretrained-reuse-r2\dcase2025-av-devtest-evaluation-20261101-attempt-01`; prediction JSONL SHA-256 `8c66b55bcc1ebbdebb732d808b52a4bb056309522e20e9081d8d4b287b8f4431`.
- Evaluation JSON SHA-256 `1bf25743806592a083a772d6e2b82753fadde163ed32df7bb7b16b613bf91c62`.
- Evaluation runner SHA-256 at execution: `a047ffb42862ec4531abeac406eebc5879ce5ac8107c8ddf19b3ecce3d06952`.
- Command: `python scripts/evaluate_dcase2025_av_readout.py --features C:\CodexArtifacts\pretrained-reuse-r2\dcase2025-av-devtest-features-20261031-attempt-01 --training C:\CodexArtifacts\pretrained-reuse-r2\dcase2025-av-readout-20261031-attempt-01 --candidate-freeze C:\CodexArtifacts\pretrained-reuse-r2\dcase2025-av-candidate-freeze-20261031-attempt-01\freeze.json --output-dir C:\CodexArtifacts\pretrained-reuse-r2\dcase2025-av-devtest-evaluation-20261101-attempt-01`, with `PYTHONPATH` set to the worktree's `src` directory. Completed at `2026-11-01T00:34:18Z`.
- Evaluation ran once on CPU in about 4 seconds. During this work, the RTX 3080 was occupied by PID 11056 (about 15.7 GiB reported); no GPU work was started.

The feature pipeline used the frozen EfficientAT audio checkpoint and VideoPrism
checkpoint recorded in `progress.json`; VideoPrism extraction ran on CPU and
audio extraction on CUDA. This readout evaluation itself used no encoder and no
GPU. A failed first CLI launch lacked the local `src` path and stopped during
Python import, before creating the output directory; the same frozen evaluation
then completed once with `PYTHONPATH=src`.

## Overall metrics

Options are fixed as `[offscreen, onscreen]`. ECE uses 15 fixed confidence bins.
The prevalence baseline uses only the dev-train class frequencies.

| Evaluation arm | Accuracy | Balanced accuracy | NLL | Brier | ECE-15 | Mean confidence |
|---|---:|---:|---:|---:|---:|---:|
| Train-prevalence baseline | 74.32% | 50.00% | 0.5701 | 0.3820 | 0.0112 | 0.7544 |
| Audio-only | 64.85% | 50.56% | 2.6181 | 0.6238 | 0.2941 | 0.9426 |
| Video-only | 72.59% | 59.38% | 1.3098 | 0.4638 | 0.2017 | 0.9276 |
| Joint | 69.38% | 59.70% | 2.6999 | 0.5568 | 0.2677 | 0.9615 |
| Joint, audio zeroed | 68.24% | 62.37% | 1.9994 | 0.5597 | 0.2633 | 0.9457 |
| Joint, video zeroed | 63.95% | 51.71% | 2.4374 | 0.6364 | 0.2952 | 0.9348 |
| Joint, audio parent-shuffled | 69.06% | 58.97% | 2.9064 | 0.5692 | 0.2744 | 0.9650 |
| Joint, video parent-shuffled | 65.87% | 51.53% | 4.2295 | 0.6381 | 0.3089 | 0.9676 |

Onscreen recall is low: 21.18% audio-only, 32.21% video-only, 39.79% joint.
The train-prevalence baseline has higher raw accuracy than all three learned
readouts. The learned readouts are severely overconfident and have much worse
NLL/Brier/ECE than that baseline. Joint scoring is more accurate and has lower
Brier than audio-only, but has worse NLL than both unimodal alternatives and
lower raw accuracy than video-only. Zeroing audio improves joint NLL; shuffling
video worsens joint metrics substantially. These interventions are diagnostic
only and do not prove a general fusion capability or causal necessity.

Paired cluster bootstrap (10,000 resamples, seed 17) found joint-minus-audio
accuracy +4.53 points and Brier -0.0670, while NLL was +0.0818. The
parent-recording 95% percentile intervals were [+1.93, +7.01] accuracy,
[-0.1111, -0.0206] Brier, and [-0.2520, +0.3914] NLL. Joint-minus-video was
-3.21 accuracy points, +0.0930 Brier, and +1.3902 NLL; all parent-cluster
intervals excluded zero in the worse direction. There are only 78 recording
clusters and 7 rooms; the room bootstrap is particularly low-power. Its
intervals include zero for joint-vs-audio Accuracy/Brier/NLL, while
joint-vs-video Accuracy/Brier/NLL remain worse. These intervals are descriptive
for this exposed development split and were not used for selection.

## Interpretation and next gate

This experiment does **not** support the claim that a lightweight trained
readout currently extracts reliable AV decision quality from the frozen
features. The apparent near-perfect dev-train fit failed to transfer cleanly to
the held-out rooms and is consistent with overfitting and/or a task/feature
mismatch. The result does not establish an architecture ceiling: this is one
fixed seed, one readout, one task, and one exposed development evaluation.

Preserve these predictions as the one-shot baseline. Do not tune this candidate
against these dev-test labels. Before another quality experiment, establish a
separate training/validation protocol with group-disjoint dev-train validation,
resolve class-conditional transfer and calibration, and admit an independent
task that demonstrably requires both modalities. Keep the existing
multimodal-quality roadmap gates unchanged.

The large encoders were frozen feature producers and are not part of the small
readout parameter count. The readout alone has 50,178 trainable parameters for
each unimodal arm and 99,330 for the joint arm; this is not evidence of a
complete sub-200M quantized product or of the requested 95–98% cross-modal
quality target.

After this one-shot run, only formatting and an unused config parse were
cleaned in the runner to satisfy Ruff; no evaluation logic or output artifact
was changed, and dev-test was not rerun.

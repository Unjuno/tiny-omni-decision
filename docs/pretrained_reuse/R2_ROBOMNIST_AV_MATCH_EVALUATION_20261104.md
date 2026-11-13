# RoboMNIST AV-match frozen evaluation — 2026-11-04

## Result

The original frozen feature cache was resumed and verified complete at 1,780 / 1,780 groups. The seed-17 train-only readout was then evaluated once on all 2,000 fixed validation rows (500 paired source units), without refitting or checkpoint selection. No sealed audit data was opened.

The joint audio-video readout scored 53.35% accuracy versus 50.00% for either single-modality control. The paired bootstrap gain was 3.35 percentage points versus each control (2,000 resamples of 500 pair units, seed 17; percentile 95% interval 1.30–5.50 points). However, the joint readout is severely overconfident and poorly calibrated: NLL 1.2494, Brier 0.7217, ECE-15 0.3074, and mean confidence 0.8335. The unimodal controls are near chance and substantially better calibrated. This is a small task-specific accuracy signal, not reliable decision quality or broad multimodal capability.

| Readout / control | Accuracy | NLL | Brier | ECE-15 | Mean confidence |
|---|---:|---:|---:|---:|---:|
| Joint AV | 0.5335 | 1.2494 | 0.7217 | 0.3074 | 0.8335 |
| Audio only | 0.5000 | 0.6953 | 0.5022 | 0.0258 | 0.5258 |
| Video only | 0.5000 | 0.7004 | 0.5068 | 0.0442 | 0.5442 |
| Joint, missing audio | 0.5000 | 0.6931 | 0.5000 | 0.0000 | 0.5000 |
| Joint, missing video | 0.5000 | 0.6931 | 0.5000 | 0.0000 | 0.5000 |
| Joint, shuffled audio within pair | 0.4665 | 1.4318 | 0.8115 | 0.3704 | 0.8335 |
| Joint, shuffled video within pair | 0.4665 | 1.4318 | 0.8115 | 0.3704 | 0.8335 |

## Reproducibility and integrity

- Source worktree commit: `8025c706ed8eb17d78a5f3bc66cf45eb1e0d64c7`.
- Frozen split manifest SHA-256: `3f4575ba41d944310e1de83cbdc68385390581723262245b97dc0d2c71db660f`.
- Feature cache: `C:\CodexArtifacts\pretrained-reuse-r2\robomnist-av-match-feature-cache-20261103-attempt-01`; final state `COMPLETE`, 1,780 groups, fixed 2 GiB free-space reserve preserved, raw media not persisted.
- Cache runtime: Python 3.11.16, NumPy 1.26.4, JAX 0.6.2 CPU, TensorFlow 2.18.1, PyTorch/Torchaudio 2.6.0+cu124; Torch CUDA was not initialized. RTX 3080 utilization and memory remained 0% / 0 MiB.
- Resume verification and extraction took 97.29 seconds and transferred 16,733,913 bytes. The two previously missing validation groups were verified/extracted; no split, feature contract, checkpoint, or extractor code changed.
- Frozen train-only readout checkpoint SHA-256: `3592bdd0cd8343f600de1654daa21d1b920392a41f8c3c71725d3f90dff4f00d` (reused unchanged; exact saved/reloaded prediction parity).
- Evaluation output: `C:\CodexArtifacts\pretrained-reuse-r2\robomnist-av-match-readout-20261104-frozen-validation-attempt-01`; status `COMPLETE`, 2,000 rows and 500 pair units, runtime 0.97 seconds, no GPU. Checkpoint hash was verified; prediction JSONL SHA-256 is `1e48a23af122b2fb6ba471f79e4a9f666baee5ce0f527a9fb772e87e5fd0aba4`.
- Evaluation config SHA-256: `6a5cb2af3c810ca97984609d79f8230190a5af268b091f14ca97f4b031f4e831`.
- Validation groups have documented metadata-only cross-task exposure; episodes are disjoint but rooms/devices/scenes are not held out. Treat this as development validation evidence, not a blind or independent estimate.
- One attempted launch from the isolated extractor environment failed before creating its output because that environment lacked Torch. The documented training/evaluation Python 3.11.9 / NumPy 2.4.6 / PyTorch 2.6.0+cu124 runtime was then used for the frozen evaluation; no weights or validation rows were changed by the failed launch.

## Interpretation and next gate

RoboMNIST AV-match now demonstrates a modest above-chance joint accuracy signal with a positive paired interval, while also exposing a severe calibration failure. It does not establish a shared general-purpose Decision model, robustness to new rooms/devices, or the requested cross-modal quality. Do not temperature-fit or retune against this validation result. Keep the checkpoint, feature cache, predictions, and this report as a frozen development record. Select the next R2 task only through the existing task-admission gates and genuinely independent data; do not promote this readout as a product artifact.

The cache and readout remain external artifacts and were not added to Git. Sealed audit data was not inspected. No legacy worktree, branch, PR, dataset, or checkpoint was removed or overwritten.

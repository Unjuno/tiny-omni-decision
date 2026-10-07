# Teacher Quality: Physion++ Clean Development v3

Status: **incomplete; safety-stopped for local disk/pagefile pressure**. This is not a quality-based rejection and is not a Teacher promotion. No sealed/final audit was loaded.

## Run definition and provenance

- Experiment: `teacher-v2-physionpp-clean-dev-v3-seed17-2048`, seed 17.
- Branch: `codex/teacher-quality-physionpp`.
- The run invocation recorded source commit `60676815f2af6489ea1325ddab0abc2d0a9d93b2`. At stop, the branch was at `a7b2f8aff3502979e278205a1945c80cf5905c16`; the live `trainer.py`, `training.py`, config, and base-manifest hashes matched the hashes saved in the step-256 checkpoint.
- Started 2026-10-06 23:24:12 UTC (2026-10-07 08:24 JST), initialized from the pinned pretrained base and a new LoRA. It did not warm-start from Teacher v1 or the prior E-long adapter. The v1 adapter was read-only reference-evaluation input.
- Base: `google/gemma-4-E2B-it-qat-q4_0-unquantized`, revision `6befbaca7398925921802abd1f277b495b78b738`; base weights SHA-256 `33fe0cece08fb527ffefbd1a3a9ce73bd71073727993a283506293e5c6bf0137`.
- Training/validation corpus SHA-256: `0235d4c20037d6e2d6281d08ad3838842a62ec537932834e069d648982c521f8` / `352928c0e31fda0f12cb914c62032583d28360900b72204510440b1aa003e694`. The frozen corpus contains 96,450 train rows and 1,304 validation rows. Preflight found zero train/validation overlap for source IDs, normalized content, media identity, and source assets.
- Validation selector: 1,304 examples, 326 each from Amazon MASSIVE (text), Clevr-4 (image), LibriSpeech (audio), and Physion++ Readout (video). Each selected modality had 326 distinct assets. Selector order SHA-256: `923073bb33eeee64d7a04ad07c6a8c3d84b8a1e1afdaf30baf31eebec1742ae7`. The video selector was all `temporal_descriptive`; results do not establish performance across all video task types.
- Effective setup: 2,048 planned optimizer updates, accumulation 4, one example per microbatch, max sequence length 1,024, 8 video frames, cosine LR `5e-5`, warmup ratio `0.03`, decoder-all-linear LoRA rank 16 / alpha 32 / dropout 0.05, rsLoRA off, encoders and projector frozen, one repeat maximum, validation/checkpoint interval 128, patience 17. Target modules were 205 actual language-model paths (q/k/v/o and gate/up/down projections); adapter held 24,158,208 trainable parameters.
- Corpus intended for 8,192 unique consumed examples, with zero planned repeats. At the last complete checkpoint, 1,024 unique examples had been consumed across 938 unique underlying assets and 263 video scenes. The saved step-256 task mix was 205 temporal-descriptive, 51 explanatory, 51 predictive, and 34 counterfactual video examples. The subsequent 128-update tail appears in training history through step 384 but is not in the resumable state and must be replayed.

## Validation observations

Metrics are from the same fixed clean-dev-v3 selector and are development results, not a blind audit. Source metrics correspond to the single named source for each modality in this selector.

| Modality / source | Step 128 Accuracy | NLL | Brier | ECE | Step 256 Accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Audio / LibriSpeech | 0.9939 | 0.0207 | 0.0060 | 0.0150 | 0.9969 | 0.0047 | 0.0023 | 0.0031 |
| Image / Clevr-4 | 0.6534 | 1.1295 | 0.4771 | 0.1111 | 0.6687 | 1.1060 | 0.4657 | 0.1844 |
| Text / Amazon MASSIVE | 0.6472 | 1.9077 | 0.5232 | 0.1507 | 0.6779 | 1.8867 | 0.5053 | 0.1818 |
| Video / Physion++ Readout | 0.4877 | 0.7167 | 0.5227 | 0.1081 | 0.4939 | 0.6882 | 0.4951 | 0.0763 |

At step 256, overall and macro Accuracy were both 0.7094; minimum modality Accuracy was 0.4939 (video), macro NLL 0.9214, macro Brier 0.3671, and macro ECE 0.1114. The selector chose step 256 with score 0.70515. The saved peak allocated VRAM was 11.92 GB. Validation took 1,916.5 seconds at step 128 and 1,927.6 seconds at step 256. Training history records step 384, but the step-384 validation metrics and predictions were not saved. Two early observations are not sufficient to infer a plateau or reject the learning-budget hypothesis.

## Safety stop and exact-resume evidence

The process was stopped after C: free space fell to approximately 0.25 GiB while the process was active. The latest complete exact-resume snapshot was 370.63 MiB, so the available space was insufficient to safely publish the next atomic checkpoint. The run output itself remained about 0.46 GiB and did not grow during the step-384 evaluation. Windows pagefile allocation increased from 11,592 MiB after the process exited to 16,305 MiB while it ran; C: free space then returned to 4.88 GiB after termination. At stop, available RAM was about 12 GiB and the observed process was still responsive; the failure condition was checkpoint-storage headroom, not a metric or numerical failure.

The process was terminated without modifying the existing snapshot. `resolve_resume_checkpoint` passed for `checkpoints/resume-step-000256-f2de92412b93410e8af2e6339f4b300b`:

- global step 256, sample index 1,024, selected step 256, with adapter, optimizer, scheduler, complete Python/NumPy/Torch CPU/CUDA RNG state, selected adapter, predictions, and selector metadata present;
- train and validation corpus hashes matched the checkpoint;
- current trainer, training, config, and base-manifest file hashes matched the checkpoint hashes;
- the exact-resume sampler order and all loaded-model checks have **not** been exercised by a resumed training process.

A future continuation must use this checkpoint, preserve the exact config/corpus/code, discard/replay the uncheckpointed updates 257–384, and write to a fresh output directory. The current runner re-evaluates the base and reference adapter before restoring training state, so that setup cost will recur. Do not resume until at least 1 GiB additional C: headroom is available while the model is active, or another local writable volume is provided. No files were deleted to make room.

Run files remain outside Git at `C:\CodexArtifacts\tqpp\physionpp-clean-dev-v3-generation-02\run-seed17-2048-attempt-02\`. A machine-readable stop record is `attempt-status.json`. The selected adapter SHA-256 is `61f985f24cef21bcf710d98c823194f72975068604556312bbf43d54c0319a5a`; it is an interim development checkpoint, not a final Teacher artifact.

## Decision

This run is **blocked/incomplete for infrastructure reasons**. It does not support a conclusion that longer training is ineffective. The fresh-audit gate remains untouched and unmet; all four modalities must still reach at least 90% on a future sealed audit.

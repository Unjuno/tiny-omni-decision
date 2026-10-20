# CLEVRER synthetic AV temporal decision readout results — 2026-10-20

## Outcome

The matched GPU run completed all four frozen candidates for 2,048 optimizer updates each (8,192 updates total). This is a **synthetic temporal-binding diagnostic**, not natural audio/video understanding or evidence that the broad 95–98% product target has been reached. No sealed audit was opened, no encoder was updated, and no quantization or product promotion was performed.

The gated joint LSTM reached a validation-selected best of 78.97% Accuracy, 0.4566 NLL, 0.3031 Brier and 0.0670 ECE at step 512. The audio-only arm's best was 72.31% / 0.5051 / 0.3515 / 0.0868 at step 256. The video-only and mean-pooled controls stayed at chance. On this task, the video sequence adds useful information to the learned audio-only arm, but the joint model does not solve the task and substantially overfits by the fixed 2,048-step endpoint.

## Frozen task and implementation

The task, source, splits and model configuration are specified in the [protocol](R2_CLEVRER_SYNTHETIC_AV_PROTOCOL_20261020.md). CLEVRER itself is silent. Each event has two paired rows with the same question and video, while an identical synthetic 880 Hz, 80 ms tone is placed before or after the named collision. The answer is which side of the collision contains the tone. A train-fit absolute-onset threshold had already reached 56.67% on validation, so audio-only learning was included as a shortcut control.

Four seed-17 candidates used the same 16-frame frozen VideoPrism features, frozen MiniLM question features, waveform-derived tone-energy features, split, sampler, optimizer and schedule:

1. `joint_lstm`: gated joint audio/video sequence readout;
2. `audio_only_lstm`: audio sequence with video masked;
3. `video_only_lstm`: video sequence with audio masked;
4. `joint_mean_pooled`: joint input with temporal order removed.

Each readout has 48,868 trainable parameters. Frozen encoder revisions and hashes are recorded in the effective config. The VideoPrism checkpoint has 114,365,184 parameters and MiniLM has 17,389,824, for 131,803,876 parameters including the readout. This is a parameter count for the unquantized frozen feature encoders plus readout, not a packaged/deployable model measurement.

## Run record and integrity

- Source commit: `82f260b0525f1f95c943dd57009821ec46b7eda1`.
- Command: `python scripts/run_with_hotgpu_guard.py --max-wait-seconds 3600 --reserve-mib 3200 --block-if-free-mib 14000 --min-child-free-mib 8000 -- python.exe -m scripts.train_clevrer_synthetic_av_temporal_decision --config configs/pretrained_reuse/clevrer_synthetic_av_temporal_decision_v1.json --feature-dir C:\CodexArtifacts\pretrained-reuse-r2\clevrer-synthetic-av-timing-features-v1-attempt-04 --feature-report C:\CodexArtifacts\pretrained-reuse-r2\clevrer-synthetic-av-timing-features-v1-attempt-04\feature-report.json --sample-dir C:\CodexArtifacts\pretrained-reuse-r2\clevrer-synthetic-av-media-v1-attempt-01 --output-dir C:\CodexArtifacts\pretrained-reuse-r2\clevrer-synthetic-av-temporal-decision-v1-attempt-01 --device cuda`.
- Effective config SHA-256: `6edbd88442f29de8917377b734bda933c8ac0a7e51af88700825be6b7cd60f4b`.
- Feature cache SHA-256: `c7382bbe66ed5d0943a658d9c251feb455478f617db9516649956cb2011ebf2b`.
- Ordered train and validation sample ID SHA-256: `df0b7525568d4ad8b9d22f536c241314cc42f218db65b4e0c0cf339c8c2532d7` and `572bf3aeb96b6b7ca102cc7ae2fa5efcc22a792f2aff97b50a840f4ebb0dbd40`.
- Each candidate consumed 130,748 examples, including all 1,596 unique train rows and 322 unique scenes; the remaining 129,152 draws were repeats. The 390-row validation set covers 79 scenes. Scene and media-hash split overlap were both zero.
- The job ran on the local NVIDIA GeForce RTX 3080 Laptop GPU through HotUGPU. HotUGPU waited for the cooperative reservation and then launched; it did not stop the other process. Python 3.11.9, PyTorch 2.6.0+cu124 and CUDA 12.4 were recorded. Peak allocated VRAM was 310,462,464 bytes for the sequence readouts and 315,372,032 bytes for the mean-pooled control.
- Start 2026-10-20 13:17:03.628818 UTC; end 13:18:13.817175 UTC; wall time about 70.2 seconds. Candidate training times were 14.88, 17.75, 15.94 and 16.00 seconds; recorded validation time totaled about 1.35 seconds. Compute cost: $0.
- The external run directory is `C:\CodexArtifacts\pretrained-reuse-r2\clevrer-synthetic-av-temporal-decision-v1-attempt-01`. It contains effective config, run metadata, four complete histories, best and final checkpoints, per-row predictions, and results. `results.json` SHA-256 is `f175a2f7ec9fa80c4780c9f3df9af867dcd18d05271c4bb5efa6d0aca1b49b18`; the run metadata says `complete` and `sealed_audit_opened: false`.
- All eight checkpoint hashes were checked against `results.json`. The selected joint checkpoint SHA-256 is `c6e010d460773972a084fff9ebe82fbddfda1554ecb2f51d3d64e89af3c20ba1`; its final-step checkpoint is separately retained as `4d53bb6ee6db5f04c1d9b1b1a64a84c14ed3279f4723adff73ec0ee7f03cecc6`.

## Learning curve

The following rows come from the fixed full-train-set evaluation and the same 390 development-validation rows at each scheduled point. The selector is minimum validation NLL. These repeated validation looks are for development, not a blind test. Values are Accuracy / NLL.

| Candidate | 512 | 1,024 | 1,536 | 1,920 | 2,048 | Best step |
|---|---:|---:|---:|---:|---:|---:|
| Joint LSTM | .790 / .457 | .777 / .506 | .785 / .672 | .787 / .890 | .785 / .943 | 512 |
| Audio-only LSTM | .721 / .535 | .731 / .547 | .708 / .579 | .664 / .701 | .669 / .686 | 256 |
| Video-only LSTM | .500 / .693 | .500 / .693 | .500 / .693 | .500 / .693 | .500 / .693 | 768 |
| Joint mean-pooled | .500 / .693 | .500 / .693 | .500 / .693 | .500 / .693 | .500 / .693 | 1,408 |

Joint training Accuracy/NLL at steps 512, 1,024, 1,536, 1,920 and 2,048 was `.848/.335`, `.900/.219`, `.953/.120`, `.991/.042` and `.981/.066`. Its validation NLL rose from .457 at step 512 to .943 at step 2,048, while ECE rose from .0670 at its selected point to .1761 at the endpoint. Audio-only validation also deteriorated after its early best (step 256), ending at .669 Accuracy and .6865 NLL. This is direct evidence of overfitting in these runs; the rolling recent training loss is not used as a generalization-gap estimate.

## Selected and endpoint metrics

All metrics below use the same 390 validation examples. ECE uses 15 bins.

| Candidate/checkpoint | Step | Accuracy | NLL | Brier | ECE | Mean confidence |
|---|---:|---:|---:|---:|---:|---:|
| Joint LSTM, selected | 512 | .7897 | .4566 | .3031 | .0670 | .8315 |
| Joint LSTM, final | 2,048 | .7846 | .9432 | .3704 | .1761 | .9533 |
| Audio-only, selected | 256 | .7231 | .5051 | .3515 | .0868 | .7870 |
| Audio-only, final | 2,048 | .6692 | .6865 | .4326 | .1551 | .8244 |
| Video-only, selected | 768 | .5000 | .6931 | .5000 | .0001 | .5001 |
| Video-only, final | 2,048 | .5000 | .6932 | .5000 | .0047 | .5047 |
| Joint mean-pooled, selected | 1,408 | .5000 | .6931 | .5000 | .0001 | .5001 |
| Joint mean-pooled, final | 2,048 | .5000 | .6932 | .5000 | .0043 | .5043 |

The selected joint model got both variants right for 58.46% of event pairs and changed its prediction across the two paired rows for 58.97% of pairs. For the within-pair audio-swap intervention, 58.97% of predictions flipped; after the swap, accuracy against the original target was 21.03%, and accuracy against the donor (opposite) target was 78.97%. This supports sensitivity to the synthetic audio timing, but not robust cross-modal reasoning by itself.

## Uncertainty and interpretation

A paired bootstrap sampled validation scenes (79 clusters), with 5,000 replicates and seed 17. For selected joint versus selected audio-only, the joint-minus-audio Accuracy difference was +6.91 percentage points, with a percentile 95% interval of +3.43 to +10.65 points. The NLL improvement was 0.0497 (interval 0.0006–0.0973) and Brier improvement 0.0490 (0.0141–0.0841). Both checkpoints were selected on this same validation set, so these intervals are descriptive and selection-biased; they are not blind confirmation.

The result supports a narrow claim: in this synthetic timing setup, the joint sequence readout uses information beyond the learned audio-only arm and beats it on selected validation. It does **not** show that natural video or audio encoders can solve a real AV decision task. The video-only arm at chance also means these data do not establish independent video understanding. The absolute-onset shortcut, repeated generated question template, source-scene screening limitation (candidate identities were not certified by hashing all historical local media), small number of 79 validation scenes, and reuse of validation for checkpoint selection all limit causal claims. The test is not representative of the product's four-modality distribution.

## Decision and next evidence

- Execution: **PASS** — all four candidates completed 2,048 updates; result, histories and checkpoint hashes are present; the selected checkpoint was reloaded and its validation metrics matched the recorded selection.
- Temporal/video contribution: **narrowly supported on this synthetic task**; selected joint beats selected audio-only, but selection bias and the synthetic cue prevent a natural-AV conclusion.
- More optimization steps: **not supported as a remedy in this run**. Joint overfit after its step-512 best, and the audio-only arm also deteriorated. This does not establish a general model-capacity or architecture ceiling.
- Product quality: **not established**; 95–98% on each modality remains unmet and unmeasured here. No product Teacher is selected.
- Next work should prioritize a rights-cleared natural synchronized AV task where each label demonstrably requires both streams, followed by a clean group split and paired modality-ablation controls. Do not spend more steps on this synthetic corpus as if that would establish natural AV capability.

The v1/v2 corpus, checkpoints and artifacts were read-only during this work. The synthetic media/feature caches and model checkpoints remain outside Git.

# Teacher Quality: Physion++ Clean Development v3

Status: **2,048 optimizer updates completed after exact resume; selected-checkpoint reload verified.** The initial attempt was safety-stopped for local disk/pagefile pressure. That interruption and a later checkpoint-serialization failure are retained below; neither changed the experiment conditions. This remains development-selector evidence, not a Teacher promotion. No sealed/final audit was loaded.

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

The initial attempt was incomplete for infrastructure reasons; the exact continuation below resolved the run-completion gate without changing its conditions. The completed run still does not establish a model-family ceiling or product quality. The fresh-audit gate remains untouched and unmet; all four modalities must still reach at least 90% on a future sealed audit.

## Exact continuation and completed learning curve (2026-10-08)

The step-256 stop above was not treated as a quality result. An exact continuation
first resumed the verified step-256 snapshot and reached step 1,280; its next
checkpoint serialization failed under local storage pressure. Its latest exact
snapshot was step 1,152. A separate continuation then passed the existing exact-
resume preflight from that snapshot and completed the same seed-17 experiment
through step 2,048. It used the same base, frozen corpus, config, source code,
validation IDs/order, sampler, optimizer, cosine scheduler, selector and RNG
state. The uncheckpointed tail after step 1,152 was replayed; no fresh seed or
hyperparameter run was started. The original attempts and files remain intact.

The durable continuation output is
`C:\CodexArtifacts\tqpp\physionpp-clean-dev-v3-generation-02\run-seed17-2048-resume-step1152-exact\`.
Its `resume-preflight.json` records matching config, corpus-manifest and code
hashes, optimizer/scheduler/adapter/RNG presence, consumed sampler order,
selected-adapter hash and validation-selector order. The preflight records
resume step 1,152/sample index 4,608, and the run summary records the resumed
run reaching step 2,048. Trainer state ends at sample index 8,192;
all 8,192 training examples were unique. The selected checkpoint is step 1,664,
and the terminal step-2,048 adapter is separate. The trainer's saved
`checkpoint_reload_verified` value is true.

The following checkpoints share the same 1,304-example development selector
(326 examples per modality). All 326 video questions are temporal descriptive;
this result cannot establish other video task types. The selector was used for
checkpoint choice, so these are not blind-test estimates.

| Step | Macro Accuracy | Minimum modality Accuracy | Macro NLL | Video Accuracy | Video NLL | Selector score |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 0.7416 | 0.4939 | 0.6748 | 0.4939 | 0.6815 | 0.43369 |
| 768 | 0.7661 | 0.5061 | 0.6749 | 0.5061 | 0.6878 | 0.42350 |
| 1,024 | 0.7722 | 0.5123 | 0.7836 | 0.5123 | 0.7899 | 0.53690 |
| 1,152 | 0.7676 | 0.5092 | 0.6108 | 0.5092 | 0.7136 | 0.35697 |
| 1,280 | 0.7661 | 0.4816 | 0.5861 | 0.4816 | 0.6939 | 0.33625 |
| 1,536 | 0.7853 | 0.5092 | 0.5527 | 0.5092 | 0.7118 | 0.28965 |
| **1,664 selected** | **0.7876** | **0.5092** | **0.5371** | **0.5092** | **0.6969** | **0.27100** |
| 1,920 | 0.7906 | 0.5276 | 0.5515 | 0.5276 | 0.7080 | 0.28096 |
| 2,048 final | 0.7922 | 0.5368 | 0.5508 | 0.5368 | 0.7078 | 0.27805 |

Selected step 1,664 macro Accuracy/NLL/Brier/ECE was
`0.7876 / 0.5371 / 0.2621 / 0.0567`. Per-modality Accuracy/NLL/Brier/ECE:

| Modality | Accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|
| Audio | 1.0000 | 0.000215 | 0.0000003 | 0.000215 |
| Image | 0.8650 | 0.4173 | 0.2003 | 0.0498 |
| Text | 0.7761 | 1.0340 | 0.3445 | 0.1158 |
| Video | 0.5092 | 0.6969 | 0.5036 | 0.0610 |

At the terminal step 2,048, macro Accuracy/NLL/Brier/ECE was
`0.7922 / 0.5508 / 0.2658 / 0.0633`. Video Accuracy increased to 0.5368, but
the fixed selector score was worse than at step 1,664, so the best checkpoint
was retained. Text and Image NLL/ECE also worsened from the selected checkpoint.
This is a validation trade-off, not permission to select the final weights.

The same-selector Teacher v1 reference had macro
Accuracy/NLL/Brier/ECE `0.7247 / 0.8932 / 0.3765 / 0.1206`; modality Accuracy
was Audio 1.0000, Image 0.7730, Text 0.6104 and Video 0.5153. The selected
candidate improves macro metrics and Image/Text Accuracy on this development
selector, while Video Accuracy is 0.61 percentage points lower than the v1
reference. The terminal checkpoint is 2.15 points above the reference on Video
Accuracy, but has worse macro NLL than the selected step. This one seed and
selection-touched selector do not prove broad transfer or 90% video quality.

At step 2,048 the run had consumed 8,192 examples with no repeats, 5,391 unique
assets, and 319 CLEVRER scenes. Source counts were CLEVRER 2,512; Physion++
Readout 219; Clevr-4 2,048; and 683 each from Typed Decisions Synth, MASSIVE,
LibriSpeech and Open-Jev, plus 682 Speech Commands. Video-task counts were 555
counterfactual, 914 explanatory, 488 predictive and 774 temporal descriptive.
The selected step-1,664 checkpoint corresponds to 6,656 consumed examples.

The selected adapter SHA-256 is
`397be9640c79c10f9a56cdf00c2e9c76da743271f3c0271c0a7386353a194ee7`; the
step-2,048 final adapter SHA-256 is
`71fa0fb1d61214c8367943315a3079e8d82dbf7f796c9b0cca22e1fa4f1aff78`.
The run manifest SHA-256 is
`9e398496a37b89ddadf9dc548627d39ec824d29d380e5678b353137b903e4cda`, training
history SHA-256 is
`5800a48fe79fb73e1a24c28711761a25637e0405d38afeccc91d3a7b6a3c79ce`, and
the final metrics/predictions SHA-256 values are respectively
`91bbb5fae93dff26fe1ee510240b13f0f3fb1a1934e260d0a7e97ec1c3ea2be3` and
`6452174009a8addb6cbafe82f66c6a085f200333146ad645c239b0ae6219a4fa`.

The source commit recorded by the run is
`740ec0a7d2d8ba870b384a130c8b0821537b9626`; config SHA-256 is
`40ac489e1f5b06b18bdd296a02993c9e80764fb8e1d442718298d3db858715d0`, effective
config SHA-256 is
`8ea1ff704071e99cc3cbfd7c04e3fd40419399a02544f406d5e0838ca1ba0b71`, and the
train/validation corpus hashes remain those listed above. The pinned base and
processor revision is `6befbaca7398925921802abd1f277b495b78b738`.

The continuation used only the local RTX 3080 Laptop GPU, BF16, PyTorch
2.6.0+cu124, CUDA 12.4, Transformers 5.6.2, PEFT 0.21.2, Accelerate 1.15.0,
Python 3.11.9 and the math SDP backend. The selected adapter is 96,693,360 bytes
with 24,158,208 trainable LoRA parameters; the complete Teacher remains a
5,128,455,712-parameter research model. Peak allocated VRAM was 11,954,854,912
bytes. The exact continuation reports 26,230 seconds wall time; the cumulative
run metadata reports 59,103 seconds including prior attempts and evaluations.
Cloud cost was $0. No sealed audit, ternary conversion, adapter merge or product
promotion occurred. The 90%-per-modality quality gate remains unmet, with
Physion++ Video the weakest modality at 50.92% for the selected checkpoint.

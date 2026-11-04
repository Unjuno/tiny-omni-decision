# CLEVRER Video/Text Controls (2026-10-11)

## Question and scope

This development-only control asks whether the selected frozen VideoPrism +
MiniLM readout uses the *correct scene's* video to answer CLEVRER explanatory,
predictive and counterfactual binary choices. It compares a previously fixed
multimodal readout with a text-only readout fit on the same frozen training
choices and with the multimodal readout evaluated after a deterministic
scene-level video derangement. This is not a sealed audit, a product promotion,
or a general video-understanding result. The new validation was used only for
final reporting; it did not select a checkpoint, epoch, temperature or
threshold.

## Data and integrity

- Source: `MIT-IBM/CLEVRER@98b842082ba4f7c18b6b9e3f39145871782a65ef`.
- The selection used seed 17 and 200 validation scenes, expanded into 2,777
  binary choices: explanatory 1,185, predictive 337, counterfactual 1,255.
- Compared with discovered prior CLEVRER sample/prediction inputs, the selected
  scenes and normalized question/options content each have zero overlap. The
  saved preflight counted 10,412 prior scene IDs and 7,502 prior normalized
  input/options keys. All questions/choices from a scene remain grouped.
- The one row in `train.jsonl` is a train-scene feature-extractor witness only;
  it was excluded from training and evaluation. The video feature cache
  contains 201 assets: the 200 validation scenes plus that witness.
- The selected 200 video files were SHA-256 checked. The upstream video archive
  is 6,206,059,321 bytes; only 32,173,790 range bytes were read, so the full
  archive SHA-256 was **not** verified. The archive ETag and pinned dataset
  revision are recorded in the fetch report.
- Sealed-audit inputs were not loaded.

| Input | SHA-256 |
|---|---|
| Ordered validation IDs | `b5fc1f6fc72fd0138bc85d85645137c54df4ad806f40d28e438cd830b7506505` |
| Validation JSONL | `1eaf8b65eb632f930b201bd9e4d48b904d69aaa49f85764298083740d6bd5ae0` |
| Fetch report | `54b3b0cd0a4fadbed7a171572ee051dd5bcd698b4e8d0d6a739ad7ad95af493f` |
| Video feature cache | `688a4f4801d439bdfd6991805cedfea2ac4530daf90206630c041cc463f618e1` |
| Video feature-cache report | `c675b05e6e8748ebc1b65b7c181f5e2093139555cd309a13822abb2450f96734` |

## Model and procedure

- Fixed video encoder: Google VideoPrism-B, revision
  `be719a406d563b66f0ac969e7c94bab8e997c81a`, upstream source commit
  `c6eba0ece3dd49a24ffc44582966997a4a59f1d8`, checkpoint SHA-256
  `61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29`.
  Eight uniformly sampled frames at 288 px were mean-pooled to 768 dimensions.
- Fixed text encoder: `sentence-transformers/paraphrase-MiniLM-L3-v2`, revision
  `4ca70771034acceecb2e72475f72050fcdde4ddc`, local weights SHA-256
  `cf1e4e2d420c664973037c3c73125d7a8fc69952495093ef8f50596f8943a433`.
- Training choices are the exact 3,794 rows used by the existing binary
  reasoning probe, from 256 scenes/assets (explanatory 1,560, predictive 352,
  counterfactual 1,882). The fresh 2,777 validation rows are scene- and
  normalized-content-disjoint from those rows.
- Existing training-choice JSONL SHA-256 is
  `b98c3089f9850fa2343c77561a963306fd0b8201912e38000143fc72da36f032`; its
  frozen experiment config SHA-256 is
  `6b545100b694ab75c16d5ecc62c25adb05140f6c0c0fe2ebb9025ea94a6d3cf9`.
- The multimodal readout is the existing `best-readout.pt` (295,169 trainable
  readout parameters), held fixed. It was selected earlier on the old
  development validation and was not refit here.
- Text-only uses the same 3,794 training rows, MiniLM features, hidden size 128,
  seed 17, AdamW (`lr=0.001`, `weight_decay=0.001`), equal question-type then
  equal scene weighting, batch size 64, and 24 epochs. It uses fixed epoch 24;
  this fresh validation did not select its checkpoint. It has 196,865 trainable
  parameters.
- Video shuffle is one deterministic seed-17 derangement of validation scene
  IDs. Every question from a scene receives the same donor scene's video
  feature; text, target, scorer and preprocessing are unchanged. The paired
  bootstrap is conditional on this one shuffle and does not estimate variation
  across alternative shuffle assignments.
- Metrics are reported example-weighted and as an unweighted macro across the
  three question types. ECE uses 15 equal-width confidence bins. Confidence
  intervals are 10,000 paired resamples of 200 scenes, seed 17, percentile
  95% intervals. They are descriptive development-validation intervals and do
  not account for model-selection history or shuffle-assignment uncertainty.

## Results

Metric order is Accuracy / NLL / Brier / ECE-15. Each task count is the number
of binary choices.

| Fixed evaluation condition | Macro across task types | Example-weighted overall |
|---|---:|---:|
| VideoPrism + MiniLM, fixed readout | `0.4990 / 0.6943 / 0.5012 / 0.0243` | `0.5067 / 0.6937 / 0.5006 / 0.0169` |
| MiniLM text-only, epoch 24 | `0.5065 / 0.6979 / 0.5047 / 0.0405` | `0.5052 / 0.6971 / 0.5039 / 0.0359` |
| VideoPrism + MiniLM, shuffled scene videos | `0.5018 / 0.6943 / 0.5011 / 0.0215` | `0.5085 / 0.6936 / 0.5005 / 0.0152` |

| Task | Count | Fixed VideoPrism + text | Text-only | Shuffled video |
|---|---:|---:|---:|---:|
| Explanatory | 1,185 | `0.5122 / 0.6932 / 0.5001 / 0.0128` | `0.5089 / 0.6966 / 0.5034 / 0.0357` | `0.5139 / 0.6931 / 0.5000 / 0.0112` |
| Predictive | 337 | `0.4748 / 0.6962 / 0.5031 / 0.0473` | `0.5104 / 0.7004 / 0.5070 / 0.0448` | `0.4807 / 0.6964 / 0.5032 / 0.0415` |
| Counterfactual | 1,255 | `0.5100 / 0.6935 / 0.5003 / 0.0127` | `0.5004 / 0.6968 / 0.5037 / 0.0410` | `0.5108 / 0.6934 / 0.5002 / 0.0119` |

Paired macro deltas are the second condition minus the fixed VideoPrism + text
condition:

| Contrast | Accuracy delta 95% CI | NLL delta 95% CI | Brier delta 95% CI |
|---|---:|---:|---:|
| Text-only − fixed multimodal | `[-0.0120, +0.0268]` | `[+0.00150, +0.00581]` | `[+0.00146, +0.00566]` |
| Shuffled video − fixed multimodal | `[+0.00028, +0.00614]` | `[-0.00019, +0.00014]` | `[-0.00019, +0.00014]` |

## Interpretation and limits

Accuracy is near chance for all three task types. Text-only and multimodal
Accuracy are indistinguishable under this paired scene bootstrap. Multimodal
NLL/Brier are modestly better than the newly trained text-only readout, but
replacing each validation video with a different validation scene produces
nearly unchanged NLL/Brier. The single shuffled assignment even has a small
positive Accuracy delta; that result is not robust evidence that shuffled
video helps because assignment uncertainty was not resampled.

Taken together, this probe does **not** show a practically useful
scene-specific VideoPrism contribution for this frozen scorer on these
CLEVRER reasoning choices. It also does not prove that VideoPrism or video
reasoning in general has no value: the task is synthetic, the scorer is small,
the multimodal checkpoint was selected on earlier development validation, the
text-only control has a separately initialized readout, and only one shuffle
assignment and one training seed were used. The next useful step should examine
label/choice construction and a prespecified temporal-sensitive control before
spending on shared-Lens training or model merging. Do not use these results to
change the sealed audit or promote a product model.

## Runtime, outputs, and reproduction

- Video feature extraction: 272.80 seconds on `TFRT_CPU_0`, 201 videos, 8
  frames, peak working set 4,168,003,584 bytes; cache size 585,641 bytes.
- MiniLM feature extraction: 2.29 seconds. Text-only readout fit: 2.52 seconds
  for 24 epochs. Three scorer evaluations: 0.18 seconds total. Paired bootstrap:
  7.53 seconds. Evaluation total: 16.97 seconds.
- Evaluation used `NVIDIA GeForce RTX 3080 Laptop GPU`, PyTorch `2.6.0+cu124`,
  CUDA runtime `12.4`; peak allocated VRAM was 129,800,192 bytes. No cloud
  compute or paid services were used.
- New artifacts are outside Git at
  `C:\CodexArtifacts\pretrained-reuse-r2\clevrer-binary-controls-{sample-v2,features-v0,results-v0}`.
  Results include the exact config, text-only checkpoint and history, all
  per-choice predictions, and report. Existing checkpoints and corpora were
  read-only.
- Evaluation config SHA-256
  `2cc0ca6388a3b5f989f6ca69a2783a6d39000b038d1380e82fce02fa3169ae09`; report
  source commit `c2d37ea93bcf33eb09b1f25d05b48b92919dc837`; evaluator SHA-256
  `4bbd58cbab698b54c5c76ecf7316e6c85301c5919fb2b2f0f9a2dbb46d618a6d`.
- Fixed multimodal readout SHA-256
  `c15f9184794717e5942ed45a2eec18e93b12ad9a15d4b27f8b3cdf4ff480be77`; its
  source report SHA-256 is
  `af755a4637d3bfea818e83e36fa505410af30e5e1b0dda4796e3e9312b4b630a`.
- Text-only readout SHA-256
  `d67516e98490936c0f61db123659a3b8b506e03335ed912958ea48c5b38b7dc2`;
  prediction JSONL SHA-256
  `251fce980966dae57733b378934e36df4712e446667b1603280898d0aeeeaffd`;
  text-only history SHA-256
  `304730e6413ef99741b4503a7dd14032637abe520647489e476085c49a5b315b`.

Commands (PowerShell; all outputs are new and external to Git):

```powershell
$env:PYTHONPATH = 'C:\CodexWorktrees\pretrained-reuse-omni\src;C:\CodexWorktrees\pretrained-reuse-omni'
py -3.11 -m scripts.prepare_clevrer_reasoning_controls --preflight-only
# The first --fetch attempt downloaded 172/200 videos before a transient range
# request disconnect. The original partial output was preserved and resumed:
py -3.11 -m scripts.prepare_clevrer_reasoning_controls --fetch
py -3.11 -m scripts.prepare_clevrer_reasoning_controls --resume-fetch

$env:PYTHONPATH = 'C:\CodexWorktrees\pretrained-reuse-omni\src;C:\CodexWorktrees\pretrained-reuse-omni'
& 'C:\CodexArtifacts\pretrained-reuse-r1\videoprism-base-r1\venv\Scripts\python.exe' `
  -m scripts.extract_frozen_videoprism_features `
  --sample-dir 'C:\CodexArtifacts\pretrained-reuse-r2\clevrer-binary-controls-sample-v2' `
  --checkpoint 'C:\CodexArtifacts\pretrained-reuse-r1\videoprism-base-f16r288.npz' `
  --source-root 'C:\CodexArtifacts\pretrained-reuse-r1\videoprism-base-r1\source' `
  --output-dir 'C:\CodexArtifacts\pretrained-reuse-r2\clevrer-binary-controls-features-v0'

$env:PYTHONPATH = 'C:\CodexWorktrees\pretrained-reuse-omni\src;C:\CodexWorktrees\pretrained-reuse-omni'
py -3.11 -m scripts.evaluate_clevrer_binary_reasoning_controls `
  --config 'C:\CodexWorktrees\pretrained-reuse-omni\configs\pretrained_reuse\path_a_videoprism_clevrer_binary_reasoning.yaml' `
  --sample-dir 'C:\CodexArtifacts\pretrained-reuse-r2\clevrer-binary-controls-sample-v2' `
  --video-features 'C:\CodexArtifacts\pretrained-reuse-r2\clevrer-binary-controls-features-v0\video-features.npz' `
  --output-dir 'C:\CodexArtifacts\pretrained-reuse-r2\clevrer-binary-controls-results-v0' `
  --bootstrap-replicates 10000
```

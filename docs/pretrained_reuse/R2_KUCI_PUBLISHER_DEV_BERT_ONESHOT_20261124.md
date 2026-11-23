# KUCI publisher Dev one-shot: Japanese BERT candidate — 2026-11-24

## Result

On the fixed official KUCI Dev split (10,228 four-choice questions), the
validation-selected first-four-layer LINE Japanese DistilBERT readout reached
**72.46% Accuracy, 0.6971 NLL, 0.3852 Brier, and 0.0175 ECE-15**. The matched
Ruri-v3-30m reference reached **40.86%, 1.2773, 0.6958, and 0.0243**. This
one-shot result strengthens the case for a compact Japanese BERT-family text
tower on this specific event-continuation task. It does not establish
multimodal performance or the project's 95–98% quality aspiration.

The official [KUCI dataset](https://github.com/ku-nlp/KUCI/tree/79afa1d63d1f262e6709e9be714d9ad5dae8d6f6)
defines a four-choice Japanese commonsense-contingency task and declares
CC BY-SA 4.0. This evaluation used only the official developer-provided
`dataset/dev.jsonl`; the model and readout were frozen before the file was
scored. The official competition test and any sealed audit were not accessed.

## Fixed comparison

Both readouts were selected previously by minimum NLL on the train-derived
seed-17 grouped validation split. This run performed inference only: no
training, temperature fitting, calibration, or candidate selection on official
Dev. Both used the same option-pair encoding, attention-mask mean pooling,
shared `hidden → 128 → tanh → 1` scorer, and softmax over the four choices.
No candidate pair was truncated. Option-order equivariance had maximum
absolute logit difference 0 for both candidates.

| Frozen candidate | Encoder parameters | Readout parameters | Dev Accuracy | NLL | Brier | ECE-15 | Mean confidence |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ruri-v3-30m | 36,705,536 | 33,025 | 40.86% | 1.2773 | 0.6958 | 0.0243 | 0.4274 |
| LINE Japanese DistilBERT, layers 0–3 | 53,912,064 | 98,561 | 72.46% | 0.6971 | 0.3852 | 0.0175 | 0.7297 |

The connected-group paired bootstrap resampled 8,857 publisher-Dev components
(event pair, normalized context, and unordered option content), 10,000 times
with seed 1729. DistilBERT minus Ruri was +31.60 percentage points Accuracy
(95% percentile interval +30.40 to +32.82), −0.5802 NLL (−0.5978 to −0.5627),
and −0.3106 Brier (−0.3206 to −0.3009). These intervals describe this public
development split and are not a sealed test result.

Agreement-stratified Accuracy was 68.22% / 73.32% / 76.52% for DistilBERT at
human agreement 2 / 3 / 4, respectively; Ruri scored 39.81% / 40.27% / 42.86%.
The same monotonic trend across agreement levels is consistent with the task's
ambiguous and difficult examples remaining challenging.

## Data integrity and limits

The pinned upstream revision is `79afa1d63d1f262e6709e9be714d9ad5dae8d6f6`.
Official Dev has 10,228 unique IDs and unique normalized question/option
content, with target counts A=2,644, B=2,479, C=2,497, D=2,608. Against the
83,127 official training rows, overlap was zero for IDs, exact normalized
question/option content, and core-event-pair groups. Normalized context alone
was shared by 976 Dev rows; 82 rows also shared a context with the separate
internal validation split. The full Dev set was retained—no rows were removed.
Intervals cluster by connected component, but context reuse limits how
independent the rows are. Possible pretraining exposure to public KUCI material
was not independently ruled out.

This is public publisher development data, not a blind or sealed audit. Freeze
these results and do not tune on or repeatedly replay this Dev split. Accuracy
remains well below 95–98%, and this is Japanese text-only evidence. Audio,
image, video, joint-modality necessity, OOD transfer, and product calibration
remain untested here.

The four-tower arithmetic in the preceding KUCI probe estimates 195,994,241
parameters when this text candidate replaces Ruri. That is a projection, not an
assembled runtime or measured package. The 111.2M-parameter Tohoku BERT-base
remains a text-teacher/reference candidate; its four-tower projection exceeds
the provisional 200M cap. The tested first-four-layer DistilBERT is the more
plausible product-budget candidate, but it still needs an integrated runtime
comparison and all non-text quality gates.

## Reproducibility

- Source commit before this evaluator was added: `87702a0bdfeca30bbc9b0fea5f8d8ae44985f3fb`.
- Exact evaluator script SHA-256 at inference time: `62bd516d94a66bb3e86f47baca66a58883ed9f5817b06a20836c613b8efacfea`.
- The post-run evaluator now labels the bootstrap rows as publisher-development data; the score inputs and calculation are unchanged. Its current script SHA-256 is recorded in the companion manifest.
- Train-derived effective-config SHA-256: `78c52c92e91b1aa92608603a59e9de170437070ef48ee48947d6ec12e87835c5`.
- Publisher Dev SHA-256: `a5205847d0c5a6a7952445a772b858e909d0fd379ea7f5b1a4effa7b1e045aa3`.
- Frozen train / validation SHA-256: `810ae90d853925a27a89a4fd23642c3898ea93d809cba0806be0487a9fb5c721` / `37c1afe8af91d5d8f8b2b2b174117be778f71c69deb82b89b7425776489f639b`.
- Selected readout SHA-256 (Ruri / DistilBERT): `43d4e19f2bed1645c1e67d87af99d24b82e1633d949e485fa92999029580f3e8` / `3164b38c54bd18b73ebddbf5c2b0edde379f7b700dbc9cf376b8a660cc28ec2a`.
- DistilBERT checkpoint SHA-256: `9d8b678bd9ce14363d5e6e3dc53c7dd7c76f8ad11394ccffd200f15e9f6819eb`; pinned revision `c32d05753d0d4be0f60482291a9452ec7e6f7d68`.
- Local RTX 3080 Laptop GPU, Python 3.11.9, PyTorch 2.6.0+cu124, Transformers 5.6.2; no cloud compute. Peak allocated VRAM was 2.09 GB (Ruri) / 1.86 GB (DistilBERT). Feature extraction plus scoring took 28.17 s / 36.61 s in the evaluator's per-candidate timer.
- Full per-row probabilities and feature-cache hashes remain outside Git under `C:\CodexArtifacts\pretrained-reuse-r2-local\kuci-temporal-decision-20261123-attempt-01\publisher-dev-once-20261124-attempt-01\evaluation-attempt-04\`. The corrected summary is `evaluation-report-corrected.json`; the original raw evaluator JSON is retained unchanged.
- Three earlier evaluation attempts stopped before producing predictions: missing SentencePiece in the default Python, an incomplete local Ruri tokenizer folder, and an overly strict inference-freeze check. They remain in separate output paths. The completed run used the existing isolated tokenizer/model environment; no dependency was upgraded.

Focused tests: 4 passed. Ruff and `git diff --check` pass. No model weights, source dataset rows, feature arrays, or prediction rows are stored in Git.

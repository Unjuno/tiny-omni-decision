# Japanese BERT-family temporal decision probe

## Result

A parameter-budgeted BERT-family text encoder is a strong candidate for the
Japanese temporal option-selection path. On the frozen train-derived KUCI
validation set, first-four-layer LINE DistilBERT with a small shared option
scorer reached **71.30% accuracy, 0.7110 NLL, 0.3896 Brier, and 0.0196 ECE**.
The matched Ruri-v3-30m baseline reached **41.26%, 1.2751, 0.6943, and
0.0191**, respectively. The validation accuracy difference is 30.04 points.

This is one Japanese text task and one seed. It is not a 95–98% result, not a
multimodal result, and not product qualification. The validation set selected
the checkpoints, so these figures are exposed development evidence. No
publisher test or sealed audit was opened.

## Why the matched-pooling run is primary

Attempt 01 stopped while evaluating its terminal readout: saving the scorer
moved the live module to CPU before its CUDA validation. The readout files
were retained; no complete result was produced.

Attempt 02 completed but used CLS pooling for both encoders. Inspection of the
pinned [Ruri-v3 model card](https://huggingface.co/cl-nagoya/ruri-v3-30m)
showed its SentenceTransformer pipeline uses mean pooling. Since CLS would
understate that reference, attempt 02 is retained as a non-confirmatory
diagnostic, not the headline comparison. It measured Ruri 40.85% / 1.2786 NLL
and LINE DistilBERT 67.79% / 0.7870 NLL.

Attempt 03 uses the same attention-mask-aware mean pooling for both encoders,
with special tokens included. The model card recommends an empty prefix for
semantic meaning; no task-specific prefix was added. Attempt 03 is the primary
comparison below. Pooling is tested directly in CPU unit tests.

## Data and split

- Source: [`ku-nlp/KUCI`](https://github.com/ku-nlp/KUCI), commit
  `79afa1d63d1f262e6709e9be714d9ad5dae8d6f6`, CC BY-SA 4.0.
- Only `dataset/train.jsonl` was used for fitting and selection. The 83,127
  records were grouped by normalized `core_event_pair`, NFKC-normalized
  context, and normalized unordered option content before a deterministic
  seed-17 split.
- Train: 74,814 examples / 59,085 connected groups. Validation: 8,313
  examples / 6,704 connected groups. Source-pair, normalized-context, and
  normalized-content overlap across these partitions is zero.
- Validation label counts: A 1,997; B 2,085; C 2,079; D 2,152.
- No publisher dev/test file or sealed audit was loaded during this run. The
  publisher dev had previously been metadata-screened and is not used for
  model or checkpoint selection.

Hashes:

| Object | SHA-256 |
|---|---|
| Source train JSONL | `77cd991d35cdc5cba23d0e2208cdc66e3173577449dffdf1503bf49e0624b076` |
| Frozen train JSONL | `810ae90d853925a27a89a4fd23642c3898ea93d809cba0806be0487a9fb5c721` |
| Frozen validation JSONL | `37c1afe8af91d5d8f8b2b2b174117be778f71c69deb82b89b7425776489f639b` |
| Assignment manifest | `7143c0c46dcfb433887ecc3e81e9b5f266e66e53bbd1a162225211fa5d9b205e` |

## Fixed experiment

- Seed 17; each candidate starts with a new readout. Encoder parameters are
  frozen and receive no gradients.
- Input: `(context, candidate continuation)` pair, four candidates per
  question; max length 512, truncate the first sequence only. No pair was
  truncated in train or validation.
- Shared readout topology: `encoder hidden size → 128 → tanh → 1`, scored
  independently for each option. AdamW, learning rate 0.002, weight decay
  0.0001, gradient norm clip 1.0, question batch 256.
- At most 12 epochs, patience 3, select lowest validation NLL with min delta
  0.001. Best and terminal readouts are separate files. Early stopping fired
  for both candidates.
- The checked-in config now describes a variable-width readout. Attempt 03's
  serialized effective-config descriptor said `768-128-1`; the executed code
  instantiated the width from each encoder's feature tensor. The saved
  parameter totals and checkpoint tensor shapes confirm Ruri used 256-wide
  inputs and LINE used 768-wide inputs. This descriptor-only discrepancy did
  not change model execution; the original effective-config hash is retained.
- All development files and weights stayed under
  `C:\CodexArtifacts\pretrained-reuse-r2-local\kuci-temporal-decision-20261123-attempt-01\`;
  no model weights, KUCI records, or features were added to Git.

Models:

| Candidate | Pinned revision / weight SHA-256 | Encoder params | Trainable readout | Total text params |
|---|---|---:|---:|---:|
| Ruri-v3-30m | `24899e5de370b56d179604a007c0d727bf144504` / `e94155e342cbfc33280c345c6e2912905fb5de6e1a3adeb938f395aafd2f8832` | 36,705,536 | 33,025 | 36,738,561 |
| LINE DistilBERT, layers 0–3 | `c32d05753d0d4be0f60482291a9452ec7e6f7d68` / `9d8b678bd9ce14363d5e6e3dc53c7dd7c76f8ad11394ccffd200f15e9f6819eb` | 53,912,064 | 98,561 | 54,010,625 |

The LINE checkpoint has six pretrained DistilBERT layers; only layers 0–3 were
loaded into the encoder. Missing-key checks were empty. The expected layers 4–5
and the unused masked-language-model head appeared as unexpected source
checkpoint keys. The four-tower estimate is **195,994,241 parameters** when
replacing the pinned Ruri text tower, 4,005,759 under the provisional 200M
pre-quantization cap. This estimate does not establish a packaged size or an
integrated four-modality runtime.

## Learning curves

Rolling train CE is an average over shuffled training batches for that epoch;
it is not a fixed-example training evaluation or a train/validation gap.

| Model / epoch | Rolling train CE | Val accuracy | Val NLL | Val Brier | Val ECE |
|---|---:|---:|---:|---:|---:|
| Ruri 1 | 1.3025 | 39.32% | 1.2921 | 0.7030 | 0.0169 |
| Ruri 2 | 1.2692 | 40.47% | 1.2881 | 0.7004 | 0.0213 |
| **Ruri 3 selected** | **1.2502** | **41.26%** | **1.2751** | **0.6943** | **0.0191** |
| Ruri 4 | 1.2343 | 41.04% | 1.2775 | 0.6946 | 0.0264 |
| Ruri 5 | 1.2215 | 41.20% | 1.2742 | 0.6928 | 0.0364 |
| Ruri 6 terminal | 1.2112 | 41.21% | 1.2779 | 0.6958 | 0.0397 |
| LINE 1 | 0.8785 | 66.53% | 0.8244 | 0.4505 | 0.0171 |
| LINE 2 | 0.7950 | 68.59% | 0.7816 | 0.4274 | 0.0112 |
| LINE 3 | 0.7501 | 69.71% | 0.7489 | 0.4099 | 0.0096 |
| LINE 4 | 0.7119 | 70.25% | 0.7393 | 0.4045 | 0.0086 |
| LINE 5 | 0.6801 | 71.35% | 0.7222 | 0.3947 | 0.0117 |
| **LINE 6 selected** | **0.6530** | **71.30%** | **0.7110** | **0.3896** | **0.0196** |
| LINE 7 | 0.6262 | 71.90% | 0.7183 | 0.3929 | 0.0333 |
| LINE 8 | 0.6017 | 72.01% | 0.7133 | 0.3895 | 0.0316 |
| LINE 9 terminal | 0.5781 | 72.02% | 0.7159 | 0.3883 | 0.0404 |

LINE's rolling train CE kept falling after the selected epoch while validation
NLL and ECE worsened at later checkpoints; accuracy and Brier slightly
improved. This is a mixed late-training tradeoff, and the frozen selector
chooses epoch 6 by NLL. Ruri's validation NLL reached its minimum at epoch 3;
the patience rule stopped at epoch 6. The results do not prove a model-family
ceiling.

## Selected validation comparison

| Candidate | Accuracy | NLL | Brier (sum over 4 options) | ECE (15 bins) | Mean confidence |
|---|---:|---:|---:|---:|---:|
| Ruri-v3-30m, epoch 3 | 41.26% | 1.2751 | 0.6943 | 0.0191 | 0.429 |
| LINE DistilBERT first 4, epoch 6 | 71.30% | 0.7110 | 0.3896 | 0.0196 | 0.726 |

Agreement-stratified Accuracy, with counts:

| Human agreement | Count | Ruri | LINE DistilBERT |
|---:|---:|---:|---:|
| 2 | 2,967 | 40.28% | 67.51% |
| 3 | 2,950 | 40.81% | 71.63% |
| 4 | 2,396 | 43.03% | 75.58% |

The paired connected-group bootstrap sampled the 6,704 frozen validation
components 10,000 times with replacement (seed 1729). Candidate minus baseline
differences were +30.04 accuracy points (95% percentile interval +28.61 to
+31.48), −0.5641 NLL (−0.5853 to −0.5430), and −0.3047 Brier (−0.3162 to
−0.2930). This interval is descriptive because validation selected the
checkpoint; it is not blind confirmatory evidence.

The scorer has no candidate-position embedding. A candidate permutation
equivariance check on up to 128 validation examples had maximum logit delta 0
for each model. Four answer labels are nearly balanced. The first-four-layer
LINE model's accuracy increased across agreement strata; difficult cases
remain substantially below the 95–98% aspiration.

## Compute and artifacts

- Local RTX 3080 Laptop GPU, 16 GiB; Python 3.11.9, PyTorch 2.6.0+cu124,
  Transformers 5.6.2. No cloud compute or spend.
- Attempt 03 ran 7m38s wall time. Feature extraction took 194s for Ruri and
  240s for LINE, including train and validation; readout fitting took 4s and
  14s. The highest encoder-call peak allocation was 337,872,384 bytes
  (322.2 MiB). The attempted-run top-level field reported only its last
  encoder call; per-call metrics are the source for this maximum.
- Each model encoded 299,256 train candidate pairs and 33,252 validation
  candidate pairs. Attempt 03 consumed 448,884 training rows for Ruri (6
  epochs) and 673,326 for LINE (9 epochs), 1,122,210 total. The selected
  checkpoints had seen 224,442 and 448,884 rows respectively. Each candidate
  saw 74,814 unique examples; repeated epochs are counted separately above.
- Best readouts and predictions, exact model hashes, per-epoch metrics, and
  effective config are in the local run directory given above. Attempt 03
  comparison report SHA-256:
  `b861faee135cc4455b1018b72926feac0da77468b298ae2374435b669754027d`.
  Effective config SHA-256:
  `f8b045df1e61d9e5a3d242d2bb8eaefbf184d62c34c9e96b63872a0288bf1b17`.
- Attempt 03 source commit: `81e3c9f52f55970ab3411650a8ebcdf6fe99c961`.
  Runner SHA-256: `92d9103e0af9447874a9a4c15fb76fcdf98c0f0dad06a0d20573334278e7d9ab`.
- Focused split/model/analysis tests: 15 passed; Ruff and diff checks pass.

## Decision and limits

**Proceed with a compact BERT-family text tower as a serious candidate.** The
matched result supports this direction for Japanese event-continuation choices
and it fits the provisional parameter cap in the four-tower arithmetic. Keep
the current Ruri path and existing BERT-base teacher/reference intact.

Next, freeze this recipe and evaluate once on a genuinely independent,
source-disjoint Japanese temporal decision set. Do not retune on this KUCI
validation. Then test the selected text tower inside the actual multimodal
runtime, including text-image/audio/video interactions and option-probability
calibration. Before any promotion, recheck exact checkpoint licensing and the
full pre-quantization parameter inventory; package size and all non-text
quality gates remain unmeasured here.

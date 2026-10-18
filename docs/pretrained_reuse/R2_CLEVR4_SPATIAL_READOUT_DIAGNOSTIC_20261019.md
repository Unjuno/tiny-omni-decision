# Clevr-4 spatial-cell readout diagnostic — 2026-10-19

## Result in one sentence

On the already-exposed 8,002-image development split, the matched 4×4 spatial
cell readout outscored the same-architecture global-pooling control by 0.80
macro Accuracy points and 0.0141 macro NLL, mainly through a 3.14-point Count
gain. This is a narrow synthetic-benchmark diagnostic, not independent or
blind evidence and not a product-quality result.

## Question and controlled comparison

The previous DINOv2 cache retained CLS plus the mean of all 256 patch tokens;
it discarded where image features occurred. Count was the weakest CLEVR-4
taxonomy. This run compared the same 180,736-parameter nonlinear cellwise
scorer with:

- **Global control:** the mean patch vector repeated into all 16 cell slots.
- **Spatial:** each slot contains one mean-pooled region from the 4×4 image
  grid. CLS, MiniLM option embeddings, scorer, tensor shapes, loss, batching,
  seed, initialization, shuffled example order, optimizer, schedule, and
  checkpoint selector were otherwise identical.

Both arms used seed 17, eight full epochs, batches of 64 image-taxonomy
questions, AdamW (`lr=0.001`, `weight_decay=0.0001`), and the earliest epoch
with minimum macro validation NLL. Each arm completed all eight epochs. The
last checkpoint was saved separately from the selected checkpoint. Both
selected checkpoints were reloaded and reproduced their saved validation
metrics and predictions exactly.

The validation split is the same internal split used by previous Clevr-4
100k readout fits and a previous fixed public-validation replay. It is
therefore already exposed development data. The spatial-vs-global result and
the bootstrap below are descriptive diagnostics; neither is a blind-test
estimate. The official public validation and sealed audit were not opened by
this run.

## Data and frozen inputs

- Source: Oxford Clevr-4 100k, archive SHA-512
  `7e8454f9f44d7b527eb20252a687b15b4cd28f3e9c79cf6147317971034c367b64bf6bba98d2a66f64b26f1e7eea2105a8acf221198342e89a844fd0007cc94c`.
- Only the 80,012 records in the official train partition were used. The
  seed-17 asset split has 72,010 train and 8,002 internal validation images.
- Unique media assets: 80,012. Train/validation ID and byte-hash overlap: 0.
  Overlap with the prior 8,424-image CLEVR-4 train manifest: 0.
- Train IDs SHA-256:
  `3f3ba7645a219f33433b2119efda16dcef723d13261cf62197ce379e4295f437`.
  Validation IDs SHA-256:
  `83361b25bb2db009c763dc5fdf3301d96e61ca5d620557be18bef53517453bd9`.
- DINOv2-small revision
  `facebook/dinov2-small@ed25f3a31f01632728cabb09d1542f84ab7b0056`, weight
  SHA-256 `ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1`,
  22,056,576 frozen parameters.
- MiniLM option encoder weight SHA-256
  `cf1e4e2d420c664973037c3c73125d7a8fc69952495093ef8f50596f8943a433`,
  17,389,824 frozen parameters.
- Cellwise candidate scorer: 180,736 trainable parameters. Encoder + option
  encoder + scorer inventory: 39,627,136 parameters. This is not a standalone
  Omni model or a quantized package.
- Spatial feature files: train `[72010,17,384]` SHA-256
  `c717518bb9215b1e2ca29ce442dedc85db8fef1b0c6f79b7f6568ccccc3485fe`;
  validation `[8002,17,384]` SHA-256
  `64e78c99eb19f6e9b3cf22731af2a4e61f502d19e97bebdce65de4851f52d350`.
  IDs preserve the exact order of the existing frozen labels. Each selected
  image was rehashed while extracted.

## Selected validation metrics

Each taxonomy has 8,002 examples. Values are macro means across the four
taxonomies, so the more frequent examples in other tasks do not dominate.

| Readout | Best epoch | Accuracy | NLL | Brier | ECE | Mean confidence |
|---|---:|---:|---:|---:|---:|---:|
| Global control | 7 | 91.57% | 0.2103 | 0.1158 | 0.0142 | 92.71% |
| Spatial 4×4 cells | 7 | 92.37% | 0.1961 | 0.1081 | 0.0140 | 93.40% |
| Spatial minus global | — | **+0.80 pp** | **−0.0141** | **−0.0077** | −0.0002 | +0.69 pp |

| Taxonomy | Global Accuracy | Spatial Accuracy | Accuracy Δ | Spatial NLL | Spatial Brier | Spatial ECE |
|---|---:|---:|---:|---:|---:|---:|
| Color | 98.58% | 98.80% | +0.22 pp | 0.0428 | 0.0198 | 0.0039 |
| Texture | 95.54% | 95.39% | −0.15 pp | 0.1574 | 0.0716 | 0.0212 |
| Count | 72.28% | 75.42% | +3.14 pp | 0.5799 | 0.3392 | 0.0291 |
| Shape | 99.89% | 99.88% | −0.01 pp | 0.0045 | 0.0020 | 0.0019 |

Spatial minus global, with 5,000 paired bootstrap replicates over image IDs
(seed 17; percentile intervals):

| Metric | Scope | Paired difference | 95% descriptive interval |
|---|---|---:|---:|
| Accuracy | Macro | +0.80 pp | [+0.54, +1.07] pp |
| Accuracy | Count | +3.14 pp | [+2.25, +4.02] pp |
| Accuracy | Texture | −0.15 pp | [−0.60, +0.30] pp |
| NLL | Macro | −0.0141 | [−0.0193, −0.0088] |
| NLL | Count | −0.0653 | [−0.0781, −0.0523] |
| NLL | Texture | +0.0107 | [−0.0046, +0.0257] |
| Brier | Macro | −0.0077 | [−0.0103, −0.0050] |
| Brier | Count | −0.0327 | [−0.0400, −0.0256] |
| Brier | Texture | +0.0039 | [−0.0023, +0.0100] |

The paired resampling unit was image ID, keeping its four taxonomy decisions
together. Because this split was repeatedly used for validation selection,
these intervals do not repair the selection exposure or constitute an
independent significance test.

## Learning curves and late-epoch behavior

Online train Accuracy below is accumulated while weights change within the
epoch; it is not a fixed-train evaluation and must not be read as a
generalization gap.

| Arm | Epoch | Train CE (online) | Train Accuracy (online) | Validation macro Accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Global | 1 | 0.4290 | 82.95% | 88.35% | 0.2788 | 0.1518 | 0.0134 |
| Global | 2 | 0.2777 | 88.52% | 89.54% | 0.2550 | 0.1388 | 0.0138 |
| Global | 3 | 0.2479 | 89.67% | 90.60% | 0.2299 | 0.1260 | 0.0103 |
| Global | 4 | 0.2334 | 90.31% | 90.02% | 0.2433 | 0.1325 | 0.0153 |
| Global | 5 | 0.2208 | 90.72% | 90.83% | 0.2190 | 0.1222 | 0.0117 |
| Global | 6 | 0.2127 | 91.15% | 90.90% | 0.2248 | 0.1237 | 0.0178 |
| Global | 7 | 0.2053 | 91.45% | 91.57% | 0.2103 | 0.1158 | 0.0142 |
| Global | 8 | 0.1985 | 91.76% | 91.51% | 0.2160 | 0.1167 | 0.0099 |
| Spatial | 1 | 0.4191 | 83.46% | 88.25% | 0.2869 | 0.1571 | 0.0149 |
| Spatial | 2 | 0.2613 | 89.26% | 90.38% | 0.2369 | 0.1307 | 0.0116 |
| Spatial | 3 | 0.2304 | 90.60% | 91.12% | 0.2160 | 0.1202 | 0.0113 |
| Spatial | 4 | 0.2143 | 91.23% | 91.27% | 0.2137 | 0.1183 | 0.0117 |
| Spatial | 5 | 0.2007 | 91.78% | 91.16% | 0.2157 | 0.1206 | 0.0116 |
| Spatial | 6 | 0.1923 | 92.13% | 91.86% | 0.2025 | 0.1127 | 0.0132 |
| Spatial | 7 | 0.1848 | 92.48% | 92.37% | 0.1961 | 0.1081 | 0.0140 |
| Spatial | 8 | 0.1774 | 92.76% | 91.43% | 0.2172 | 0.1176 | 0.0212 |

For both arms the online train metrics continued to improve while the
validation curve wobbled. Spatial validation Accuracy peaked at epoch 7 and
fell 0.94 points at epoch 8; NLL worsened by 0.0210 and ECE by 0.0072. The
best-checkpoint selector correctly kept epoch 7 rather than replacing it with
the final weights. This is a late-epoch overfitting warning on a reused
development split, not evidence of behavior on an untouched test.

## Context against the previous readout

The earlier optimizer-matched 147,713-parameter scorer on the same split
scored 90.10% macro Accuracy, NLL 0.2478, Brier 0.1331 and ECE 0.0120;
Count Accuracy was 67.48%. The spatial scorer is higher on this reused
validation (92.37% macro; Count 75.42%), but it has a different 180,736-
parameter readout architecture. This is a useful development comparison, not
a clean causal estimate of architecture versus spatial representation.

Texture is the trade-off: its Accuracy is 0.15 points lower than the matched
global control and its NLL/Brier are higher. Count improves substantially but
is still far below the project’s provisional 95% task goal. No taxonomy meets
the requested 95–98% general multimodal product target as a whole, and this
image-only synthetic benchmark says nothing about video, audio or text.

## Execution, artifacts, and reproducibility

- Source branch: `codex/r2-independent-joint-task`; runner source commit:
  `924b30e42b4b5068e665994d57618c29133d30ed`.
- Feature cache:
  `C:\CodexArtifacts\pretrained-reuse-r2\dinov2-clevr4-100k-spatial-cell-features-v1`.
- Paired readout run:
  `C:\CodexArtifacts\pretrained-reuse-r2\dinov2-clevr4-100k-spatial-readout-attempt-01`.
- Effective command:
  `python scripts/run_with_hotgpu_guard.py --max-wait-seconds 3600 --reserve-mib 3200 --block-if-free-mib 14000 --min-child-free-mib 8000 -- python scripts/train_dinov2_clevr4_100k_spatial_control.py --features <spatial-cell-features-v1> --index <clevr4-100k-index-seed17-v0> --text-model <path-a-audio-multiquery-runtime-bundle-v2/models/text-minilm-en> --output <spatial-readout-attempt-01>`.
- Python 3.11.9; PyTorch 2.6.0+cu124; CUDA 12.4; Transformers 5.6.2;
  NumPy 2.4.6; Pillow 10.4.0. Device: NVIDIA GeForce RTX 3080 Laptop GPU,
  16,384 MiB. HotUGPU reserved 3,200 MiB. Extraction peak allocation was
  317,897,216 bytes. Readout peak allocation was 332,437,504 bytes.
- Feature extraction took 892.4 s. The two complete fits including validation
  took 386.98 s (global) and 358.66 s (spatial); total measured GPU work was
  1,638.0 s (27m18s), with $0 cloud spend.
- Per arm: 288,040 unique train image-taxonomy decisions; 2,304,320 consumed
  decisions and 36,008 optimizer updates over eight epochs. The selected
  epoch-7 checkpoint had consumed 2,016,280 decisions. These counts distinguish
  total-run work from selected-checkpoint work.
- Selected checkpoint hashes: global
  `3c109e2c027ea20e8177d6f5daf30cd282a9d44e5e73356423bdc62046a41a8f`;
  spatial
  `22fdd1e72e2ffac1efe31cc4e172358fb78431aa06aac23390604bd8e0f16134`.
  Both are 725,660-byte readout-only files. They are retained outside Git.
- `config.json` SHA-256:
  `b4b8f0be2dc4da0b3e976a1fdc92ce0725a175cdb5d20a2f33a1bd767eec93be`;
  `result.json` SHA-256:
  `a6acad86fba04fb8feebf9830c4e60f4062641fb848ec3bee10d8144de29be7d`;
  paired bootstrap receipt SHA-256:
  `2385380a626b887d5595d261f8acc83b81ae3f83aac0e653a5fbecde7ec19ac7`.
- The run did not open the official public validation partition or sealed
  audit, change the old readout/features/checkpoints, quantize weights, or
  promote a model. The successful 256+256 extraction smoke is in
  `dinov2-clevr4-100k-spatial-smoke-05`; the successful one-epoch paired smoke
  is in `dinov2-clevr4-100k-spatial-train-smoke-03`. They are separate from
  the full run and are not quality results. Earlier wrapper/import/argument
  failures did not produce usable cache data; partial train-smoke folders 01
  and 02 were preserved and were not used by the full run.

## Decision

The controlled diagnostic supports retaining spatial cell features as a
promising image-path direction for Count on this development task. It does not
establish that spatial pooling alone caused the entire gain, because the
previous global model used a different scorer, and the reused validation split
was used repeatedly for selection. Texture has a small opposing trade-off,
and the epoch-8 decline signals that more training is not automatically
better.

Do not promote or quantize this scorer. Keep the current checkpoints and
validation protocol fixed. The next quality claim requires a genuinely
independent, asset-grouped image task/split, or a different unexposed task; do
not revisit the Clevr-4 public validation or sealed audit for tuning.

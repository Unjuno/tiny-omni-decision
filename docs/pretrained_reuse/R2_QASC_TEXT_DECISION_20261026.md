# QASC frozen-feature text Decision probe (2026-10-26)

## Scope and claim

This is one development-only, text-only, eight-choice science sentence-composition
probe. It does not measure broad language understanding, the four-modality product,
multimodal fusion, or the 95–98% product objective. The official QASC test shard was
not fetched, read, or evaluated. No sealed audit was used.

The model is pinned `sentence-transformers/paraphrase-MiniLM-L3-v2` at
`4ca70771034acceecb2e72475f72050fcdde4ddc`; the encoder was frozen. The only trained
component is the shared candidate readout (196,865 parameters). Inputs contain only
the source question and one supplied option. QASC fact fields are excluded from model
inputs; the unordered normalized `fact1`/`fact2` pair is used solely as a leakage and
bootstrap group identity.

## Source and split controls

- Official repository: `allenai/qasc@c44a320c2bb682d614a675d2ba4be072150db4eb`.
- Prepared-data mirror: `allenai/qasc@a34ba204eb9a33b919c10cc08f4f1c8dae5ec070`.
- License: CC-BY-4.0; attribution and Khot et al. (AAAI 2020) citation are retained.
- Train shard SHA-256: `b9a297b5ab55f1605c7682ffbb7042c26d7ecb9ff1e1aa5a820d4e791c8302d1`
  (8,134 rows).
- Validation shard SHA-256: `d1ae34ae13c5fce2c55305372c203c6cdb789728d0d7e5ea2956d55bc33f40ae`
  (926 rows).
- Prepared train JSONL SHA-256: `d2e220510f98e19a2df5e44f89d29e1b1bd5b218397264b08ff1102ea3903e51`;
  sample ID order hash: `530c89cc92a9925eb3b155610bc7dd40bbf237815b2cded33923794949341985`.
- Prepared validation JSONL SHA-256: `e567639f9c869fad6473354ca0e7ccfe001fc24f933193cb686a3a64695b95e2`;
  sample ID order hash: `5ea477bed19f0f3ec2daf346edf01784eeae78e279014f46ae7af43099bb0dd1`.
- Train/validation overlap audit: 0 shared row IDs, 0 normalized question strings,
  0 question-plus-option contents, and 0 unordered fact-pair groups. There are
  7,139 train and 780 validation fact-pair groups. Individual evidence facts occur
  across splits (63 shared fact strings); rows were grouped by the two-fact
  reasoning pair, not by any single fact. This is a limitation of the grouping
  policy and must not be described as complete evidence-fact disjointness.
- Source data contain 598 repeated normalized train questions and 33 repeated
  normalized validation questions. No example was removed. Train has 390 rows
  whose options collide under NFKC/casefold/alphanumeric normalization; 20 of
  these have the target inside a collision. Original option strings/order were
  retained. Validation has no such option collisions.
- Metadata-only controls: uniform expected Accuracy 12.5%, NLL 2.07944, Brier
  0.875; train answer-position prior Accuracy 11.23%, NLL 2.08049, Brier 0.87526.
  Shortest/longest choice heuristics were also recorded before fitting and were
  poor. Full receipt is external to Git.

## Fixed run

- Source code base: PR #15 branch `docs/pretrained-reuse-roadmap-20261010`,
  source commit `e7380a69c93b5e5fea8bbeb4252850c58861feaf` plus the uncommitted
  QASC changes in this report.
- Seed 17; 8 epochs; AdamW learning rate 0.001, weight decay 0.0001; batch size
  64 questions; no option-order augmentation; validation minimum-NLL checkpoint
  selection. These match the existing frozen text-probe runner, with its generic
  dataset mode enabled.
- Command used `scripts/train_frozen_text_probe.py --adapter generic` with the
  pinned train/validation JSONL, separate QASC manifests, local MiniLM snapshot,
  `--seed 17 --epochs 8 --batch-questions 64 --feature-batch-size 64`, and a fresh
  `run-01` output directory. It was launched behind `scripts.run_with_hotgpu_guard`.
- Run start/completion: 2026-10-26 local time; the exact wall-clock timestamps were
  not captured and are UNKNOWN. One attempted launch failed before model loading
  because `PYTHONPATH` was absent; the guarded retry succeeded. No training process
  overlapped the competing process; the HotGPU guard waited for its cooperative
  yield and then reserved 2,471 MiB.
- Effective-config SHA-256:
  `b3ffd810c655a7ac28ad9103cd405c003bbb6c7120eba8b1081dc04a6cc50fc6`.
- The successful run command was:

  ```powershell
  $env:PYTHONPATH='C:\CodexWorktrees\pretrained-reuse-roadmap-vistouch-20261026\src;C:\CodexWorktrees\pretrained-reuse-roadmap-vistouch-20261026'
  & C:\Users\junny\AppData\Local\Programs\Python\Python311\python.exe -m scripts.run_with_hotgpu_guard --max-wait-seconds 3600 -- C:\Users\junny\AppData\Local\Programs\Python\Python311\python.exe C:\CodexWorktrees\pretrained-reuse-roadmap-vistouch-20261026\scripts\train_frozen_text_probe.py --adapter generic --task-claim "QASC eight-way text-only science sentence-composition development probe; no broad text quality claim." --train C:\CodexArtifacts\pretrained-reuse-r2\qasc-path-a-seed17-attempt-01\train.jsonl --validation C:\CodexArtifacts\pretrained-reuse-r2\qasc-path-a-seed17-attempt-01\validation.jsonl --manifest C:\CodexWorktrees\pretrained-reuse-roadmap-vistouch-20261026\manifests\candidates\qasc.yaml --validation-manifest C:\CodexWorktrees\pretrained-reuse-roadmap-vistouch-20261026\manifests\candidates\qasc-validation.yaml --model C:\CodexArtifacts\pretrained-reuse-r2\models\text-minilm-en --revision 4ca70771034acceecb2e72475f72050fcdde4ddc --output-dir C:\CodexArtifacts\pretrained-reuse-r2\qasc-path-a-seed17-attempt-01\run-01 --context-policy legacy --option-max-length 256 --seed 17 --epochs 8 --batch-questions 64 --feature-batch-size 64
  ```
- Effective device: local NVIDIA GeForce RTX 3080 Laptop, CUDA 12.4, PyTorch
  2.6.0+cu124, Transformers 5.6.2, Safetensors 0.8.0. CPU smoke verified the frozen
  encoder, finite scorer output, and readout save/load before the GPU fit.
- Frozen feature extraction: 6.58 s; readout fit: 9.76 s; peak allocated GPU
  memory: 194,143,232 bytes. The 447,861,158-byte frozen feature cache is retained
  outside Git. No cloud compute or cost.
- Dataset presentations: 65,072 (8,134 unique train examples presented over
  8 epochs); 7,139 unique fact-pair groups. Validation: 926 rows / 780 groups.
  The fixed 64-question batching yields 127 optimizer updates per epoch, 1,016
  total. This is a small linear-head fit, not a base-model fine-tune.
- A PyTorch warning stated `expandable_segments` is unsupported on this Windows
  platform. The run completed; this allocator setting had no demonstrated impact.

## Results

| Epoch | Train CE (rolling batches) | Validation Accuracy | NLL | Brier | ECE-15 |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.9743 | **21.81%** | **1.9925** | **0.8521** | **0.0294** |
| 2 | 1.8621 | 20.41% | 2.0176 | 0.8613 | 0.0543 |
| 3 | 1.7387 | 17.60% | 2.1235 | 0.8927 | 0.1296 |
| 4 | 1.5577 | 17.71% | 2.2591 | 0.9373 | 0.2022 |
| 5 | 1.3644 | 17.28% | 2.3760 | 0.9676 | 0.2407 |
| 6 | 1.1754 | 17.17% | 2.6047 | 1.0157 | 0.2912 |
| 7 | 0.9993 | 16.41% | 2.7669 | 1.0491 | 0.3248 |
| 8 | 0.8321 | 16.09% | 2.9282 | 1.0865 | 0.3680 |

The validation-selected checkpoint is epoch 1. Against the theoretical uniform
eight-choice expectation (12.5% Accuracy, NLL 2.0794, Brier 0.875), the selected
readout is better on these development IDs, but the absolute result is weak. The
gap between falling rolling train CE and deteriorating validation NLL/Brier/ECE
is a strong overfitting signal. Rolling batch CE is not a fixed-train-set metric
and must not be presented as a train/validation generalization gap.

A 10,000-replicate bootstrap resampled the 780 validation fact-pair groups
(seed 17). Percentile 95% intervals: Accuracy 18.88–24.73%; NLL 1.958–2.028;
Brier 0.8417–0.8624; ECE-15 0.0190–0.0630. This describes uncertainty on the
development split only; it is not an independent test or a blind-audit result.

## Accounting correction and artifacts

The first run report counted unique row IDs as unique source groups because the
generic dataset normalizer discarded an explicitly supplied `source_record_id`.
The model inputs and predictions were not affected, but the run report's original
group-count fields were wrong. Source normalization was corrected and a regression
test added. Group counts above and in `qasc-group-bootstrap.json` were recomputed
from the frozen prepared JSONL and saved predictions. The run was not retrained;
the model result is unchanged. The original report is retained as an execution
receipt, with a separate reconciliation receipt to identify this correction.

External run directory:
`C:/CodexArtifacts/pretrained-reuse-r2/qasc-path-a-seed17-attempt-01/run-01/`

- Best readout SHA-256: `e81d27a0042c15922fa0119dc43811fcaa9387355a4d24f39d3c97ceea343068`.
- Frozen feature cache SHA-256: `997029c92fe486a2279595da994847a2c0d5442649e667cbf5e4db6b56fcad16`.
- Validation predictions SHA-256: `b54508e82d403331a88df89405637ce8b32392e0a78f9ea08032309c1db2f531`.
- Bootstrap receipt: `qasc-group-bootstrap.json` (SHA-256
  `46ed5b88894da548e176dbb2762b21f95020c98ea01fca38fb75ad046663a457`).
- Original execution report SHA-256:
  `f3b58d2bc3fe749d2a8db62257968b170391655875036100a52bcbb5c35694c5`.
- Reconciled report SHA-256:
  `5685e425167356ecda92759890d050d71815e6ae66116f770a9b8ddfa1c22fbc`;
  reconciliation receipt records both hashes and the reporting-code fix.

The generic runner now preserves group identities and emits corrected accounting
for future runs. No QASC model weights, cache, or media are added to Git. This probe
does not justify a shared multimodal Lens, weight merging, or product promotion.
Next research should diagnose task/data and learning-dynamics limitations using
new independent text tasks before changing model architecture; do not keep tuning
on this exposed validation set.

## Repository verification

- `py -3.11 -m pytest -q`: 2,474 passed, 11 skipped.
- `py -3.11 -m ruff check .`: passed.
- Training and evaluation candidate catalogs: passed validation (9 / 11 sources).
- `git diff --check`: passed.

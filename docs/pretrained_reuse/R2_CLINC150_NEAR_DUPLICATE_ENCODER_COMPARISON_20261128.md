# CLINC150 near-duplicate split and compact encoder comparison — 2026-11-28

## Decision

The corrected CLINC150 development split still supports a high score for this
narrow closed-set intent-classification task. On identical validation rows,
frozen MiniLM-L6 outperformed frozen Google BERT-Mini by 5.11 Accuracy points,
with better NLL and Brier. This does **not** establish a BERT-family ceiling,
general supplied-choice Decision quality, or multimodal quality. It does mean
that a broad BERT-size sweep on this same CLINC150 split is not the next useful
experiment. Keep BERT as a task-specific candidate/teacher and prioritize
quality-controlled data for the actual target tasks.

The earlier MiniLM-L6 95.72% score was measured on an exact-content-grouped
split. A follow-up found near-duplicate pairs across its train and validation
groups. A new split was then created from the same pinned publisher train and
validation records, grouping exact and near-duplicate text before assigning
groups. This new validation is still development data, not a blind test.

## Frozen data and grouping protocol

- Source: `clinc/clinc_oos`, revision
  `155b9c710419136e17307b80d0a13e68cd46b4ec`, `plus` configuration,
  CC-BY-3.0. Only the pinned publisher `train` and `validation` records were
  pooled; the publisher test and sealed audit were not loaded.
- Input split manifest: SHA-256
  `229a890f1921ebd5bd50de31ef704baef733d6b4c44d6ee6396ad82fc4078c6d`.
- Combined records: 18,350. All unique records were retained.
- Exact normalization: NFKC, casefold, collapsed whitespace.
- Near-duplicate candidate rule: token-set Jaccard `>= 0.80`; candidates had to
  share a lexically sorted adjacent unique-token bigram with pooled document
  frequency `<= 80`.
- All qualifying edges and normalized exact duplicates were joined by connected
  components, including cross-label matches. Components were deterministically
  assigned with seed 17, stratified by their sorted label tuple. The rule and
  implementation are in
  [`prepare_clinc150_near_duplicate_split.py`](../../scripts/prepare_clinc150_near_duplicate_split.py).
- The scan examined 588,065 candidate pairs and joined 2,490 near-duplicate
  pairs into 16,504 groups; the largest group contained 21 rows. Thirty-two
  groups contained multiple intent labels. The 80/20 split contains 14,689
  train rows / 13,203 groups and 3,661 validation rows / 3,301 groups. All 151
  classes occur in both splits; train support ranges from 80 to 280 and
  validation support from 19 to 70. The 70-row `oos` class is the largest
  validation class.
- Train/validation group overlap is zero. Five cross-label groups (23
  validation rows) fall in validation and remain visible as label ambiguity;
  they were not removed. The source rows contain no speaker, author, or session
  identity, so lexical grouping does not prove source-identity independence.
- Train JSONL SHA-256:
  `61bb3beea1648128dfebbfc462eb51bec50d513656d15156fce867d14980f9ce`.
  Validation JSONL SHA-256:
  `09ff5b99a566c17938223af0930cdcef52396d9bc945857a7af1a94b72ef4321`.
  Split manifest SHA-256:
  `2ffb8579ebb3923775238b560f9cdfde50b5ad1f633e41e0b6ed9043d13d99fc`.

## Matched model protocol and results

Both candidates used seed 17, CPU float32 features, a frozen encoder, mean
pooling over non-padding tokens, train-only per-dimension standardization, a
single linear 151-way classifier, AdamW (`lr=0.01`, `weight_decay=0.01`),
batch size 256, 60-epoch maximum, patience 6, minimum validation multiclass NLL
checkpoint selection, and no calibration. Both used the same train and
validation IDs and max length 128. No row was truncated. The two separate runs
used new random readouts; the existing CLINC artifacts were not overwritten.

| Frozen encoder + linear readout | Encoder params | Trainable readout | Selected epoch / completed | Accuracy | Macro recall | Macro F1 | NLL | Brier | ECE-15 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MiniLM-L6 | 22,713,216 | 58,135 | 7 / 13 | 0.9626 | 0.9661 | 0.9650 | 0.1621 | 0.0594 | 0.0103 |
| BERT-Mini | 11,170,560 | 38,807 | 8 / 14 | 0.9115 | 0.9148 | 0.9151 | 0.3337 | 0.1317 | 0.0113 |

The BERT-Mini checkpoint is pinned to
`google/bert_uncased_L-4_H-256_A-4`, revision
`387825ce42dbb39b87911cdf8e383ee3b25184f8`, with an Apache-2.0 model-card
claim recorded by the R1 receipt. MiniLM-L6 is pinned to
`sentence-transformers/all-MiniLM-L6-v2`, revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, Apache-2.0. All model-file hashes
are checked by the run configs before feature extraction.

The paired group bootstrap resampled all 3,301 validation near-duplicate
groups with replacement, 10,000 repetitions, seed 20261128. MiniLM-L6 minus
BERT-Mini deltas were `+0.0511` Accuracy (95% percentile interval
`[+0.0415, +0.0603]`), `-0.1716` NLL (`[-0.2049, -0.1380]`) and `-0.0724`
Brier (`[-0.0843, -0.0604]`). These intervals are descriptive only: both
models' checkpoint epochs were selected on this same validation set, so this
is not blind-test inference or a confirmatory model-selection result.

Both models' weakest intent was `oos`: MiniLM-L6 0.671 (70 rows), BERT-Mini
0.600 (70 rows). This heterogeneous unknown-intent class and the 32 cross-label
near-duplicate groups are data/task limitations to retain in future analysis,
not grounds to remove difficult examples. Per-intent results and bootstrap
draws are preserved in the external comparison artifact.

## Artifact and code receipts

The runs used source HEAD `64a499ee877d2d651b5cd2801cc0a72120cc9151`; the
runner/config files were uncommitted at execution time. Their exact SHA-256
hashes are recorded below and are included in the commit containing this report:

- Split builder: `dfecc5749c423bb0614cdfe68589e6a8f62750fe4e3b3d8623ea6be43be8b77a`.
- Training runner: `1ebb176f5d320a0f5fda40b78baab567ed8aba662c4d394bb5c3337a40c76306`.
- Checkpoint verifier: `4b0aadaae1ee2213bb039e86cd838a28687baf3691f48714d2c4f9e6f6e0cec9`.
- Paired comparison script: `82bb2f80f650ceb5c826bb8d2021f0127754778ba9d76f33c8da70d9523c3f2d`.
- Split tests: `cba1cac6127fe6cf0efcebad83a41de3373e74b2e2581c5801c1e53fc976080e`.

- MiniLM-L6 config: `e1fcf7c7ea0d6fe4de940e3049fd44804a0935c581ca8f119c8965aca3127a6c`.
- BERT-Mini config: `05b3418548de9d88d6f83a00ee751ab530b44dd29d0c70698d9b1834c6bdd5d3`.
- Split manifest: `2ffb8579ebb3923775238b560f9cdfde50b5ad1f633e41e0b6ed9043d13d99fc`.
- Paired comparison JSON: `68e53a402296d79f745ecbe02cbcaeedd100311aa5ae3719a7da9b8cb721053c`.
- MiniLM-L6 run metadata / reload verification: `c79ed3a9039ec90f4794c5336ce4e8003c7ddb05135a6450a3a7837e3e4d1ff6` / `5869cf2b5c107d5bec381ef75bc9fe6cd6ec9b487431f20329c79ae15c478afb`.
- BERT-Mini run metadata / reload verification: `7ad538ce66bed0c3c21ec8897d22b228854973442fb07c86854c9af23c856bde` / `1d9c8105174c579a22d5de3b0b3de19ad9a36f633c4e12fb746969a3d92a6203`.

The comparison was run twice from saved predictions and produced identical
JSON SHA-256 values, which guards the paired-bootstrap calculation against an
accidental one-off result.

External run roots (not added to Git):

- `C:\CodexArtifacts\pretrained-reuse-r2-local\clinc150-plus-near-duplicate-grouped-development-v1-attempt-01`
- `C:\CodexArtifacts\pretrained-reuse-r2-local\clinc150-near-duplicate-minilm-l6-frozen-readout-v1-attempt-01`
- `C:\CodexArtifacts\pretrained-reuse-r2-local\clinc150-near-duplicate-bert-mini-frozen-readout-v1-attempt-01`
- `C:\CodexArtifacts\pretrained-reuse-r2-local\clinc150-near-duplicate-encoder-comparison-20261128-attempt-01`

Each readout was independently reloaded using `torch.load(weights_only=True)`.
Sample IDs, targets and predictions matched; maximum absolute probability
error was zero; all six recomputed validation metrics matched the run metadata
exactly. Frozen encoder state hashes matched before and after feature
extraction. No official test or sealed-audit rows were accessed. These were
CPU-only runs (Python 3.11.9, PyTorch 2.6.0+cu124, Transformers 5.6.2, 14 torch
threads), taking 29.91 seconds for MiniLM-L6 and 14.64 seconds for BERT-Mini.

## Interpretation and next step

The near-duplicate grouping did not collapse the CLINC score; MiniLM-L6 remains
strong on this narrow intent taxonomy. Therefore the previous 785 cross-split
near-duplicate pairs were a real split weakness but do not, by themselves,
explain the high score. The controlled comparison favors MiniLM-L6 over this
smaller BERT-Mini on CLINC, while BERT-Mini remains a lower-cost candidate for
other text tasks. It does not justify probing larger BERT variants on CLINC.

Do not treat this result as a supplied-choice benchmark, a multimodal result,
or evidence for 95–98% generic Decision quality. The next useful text-encoder
comparison should use a newly admitted, rights-cleared task aligned to supplied
options and source-level held-out evaluation. Continue prioritizing dataset
provenance, source identity, label ambiguity, and split integrity before model
size changes. Preserve these development predictions; do not tune against the
publisher test or a sealed audit.

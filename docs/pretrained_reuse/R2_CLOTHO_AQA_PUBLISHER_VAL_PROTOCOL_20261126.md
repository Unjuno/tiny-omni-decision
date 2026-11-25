# Clotho-AQA publisher validation protocol — 2026-11-26

**State: preregistered, not yet evaluated.** This is one external development
validation of already-selected checkpoints. It is not a sealed audit, a new
training run, or a product-quality claim. The official test CSV remains
unopened and out of scope.

## Question and frozen comparison

Do the already-selected audio-plus-question scorer and its matched
question-only control transfer from the custom uploader-grouped split of
official Clotho-AQA train to the publisher's validation partition, and does
adding the frozen audio representation improve the result on identical
eligible examples?

The only compared checkpoints are:

| Condition | Checkpoint SHA-256 | Normalization SHA-256 | Frozen temperature | Existing selector |
|---|---|---|---:|---|
| Audio + question | `ac3fc2012c146538537a293e007a386fd1c738ba6ee0a7452cc89ccf05daab4e` | `f152042f1f1ff03a4affb3381b5419dfe390edf64420dbd1ed178e835bbe1791` | `1.4142135623730951` | earliest minimum raw internal-validation NLL; epoch 7 |
| Question only | `fa98ddcebc3c0175d17794ccaabb820b03d5fc600740cb3c4deaa77419af4df3` | `9c7b39b9945baaa091aa39f512b3a68d099ed0d68237bc2e89eba04df7a96850` | `1.4142135623730951` | earliest minimum raw internal-validation NLL; epoch 8 |

Both checkpoints previously used the same internal validation ID order (SHA-256
`273197786d3eba4b88b8394d5ac807e08243ad74f1912858123ee7a6db2868a6`). Their
original best weights, normalizers, temperatures, and code remain read-only.
No refitting, recalibration, checkpoint reselection, prompt changes, or
threshold selection is permitted on publisher validation.

## Rights and cohort gates

The immutable source is Zenodo record `6473207` (DOI
`10.5281/zenodo.6473207`). The official validation CSV is 428,232 bytes with
publisher MD5 `41b8f6d906a57d7a5a4ee02bbcdefcc9`. Its questions and answers are
covered by the record's MIT `LICENSE.txt`; audio rights are separately listed
per clip in `clotho_aqa_metadata.csv`.

Before any validation label is scored, freeze an inclusion manifest using only
the fixed rules below:

1. Include only exact per-clip `CC0` records. CC-BY 3.0 and all other or
   ambiguous entries are excluded because they are outside the repository's
   configured data-license allowlist. Do not infer media permission from the
   question CSV license.
2. Require valid binary votes (exactly three annotator answers). Use majority
   vote as the target; retain and report disagreement. Do not select unanimous
   examples or filter on confidence.
3. Exclude a whole publisher-validation uploader/source group if it overlaps
   any uploader, source ID, or filename used in the existing custom train,
   calibration, or validation partitions, or the audited FSD128 training
   source IDs/uploaders.
4. Exclude exact media/waveform duplicates of any previously used Clotho or
   FSD128 item, using canonical decoded mono/32-kHz waveform SHA-256. Treat
   normalized question text as a diagnostic only: repeated generic question
   wording on a different, non-overlapping audio source is not duplicated
   observation content. The content identity is the audio source plus its
   normalized question. Exclude any duplicated source-question identity as a
   whole source group. Preserve every remaining eligible record without
   difficulty or class-based filtering.
5. Require the same included question IDs, source mapping, audio features, and
   order for both checkpoints. Do not open or fetch `clotho_aqa_test.csv`.

If these gates leave fewer than 200 independent uploader groups, publish the
counts and treat results as descriptive only; do not claim a population-level
quality pass. The 200-group floor is provisional, not a formal power analysis.
If no usable cohort remains, stop before feature extraction and record the
reason.

EfficientAT's pretraining item-level source list is unavailable; overlap with
publisher validation cannot be ruled out. Report this as a limitation, even if
all observable split and waveform checks pass.

## One-shot evaluation and outputs

After gates pass, evaluate each frozen checkpoint exactly once on the same
fixed eligible cohort. Retain per-example IDs, majority target, all three
votes, raw logits, probabilities, and predictions. Report Accuracy, NLL,
Brier, ECE-15, balanced Accuracy, mean confidence, class counts, disagreement
rate, and the number of independent uploader/source groups. Compare paired
accuracy and NLL differences with a 10,000-replicate uploader-cluster bootstrap
(seed 17, percentile 95% interval). This is a descriptive paired comparison;
the external validation becomes exposed after this evaluation and must not be
used for future tuning.

Use a new output directory only:
`C:/CodexArtifacts/pretrained-reuse-r2/clotho-aqa-publisher-val-20261126-attempt-01/`.
Save the downloaded CSV and metadata hashes, inclusion manifest, excluded
group reasons, feature hashes, exact command/environment, predictions,
metrics, bootstrap seed/method, and a terminal result. No artifacts are added
to Git.

## Existing evidence and current gate

The development-only models were fit on the custom split of official train
(2,312 train / 550 calibration / 560 validation questions, grouped by
uploader). On that already-exposed internal validation, question-only scored
64.82% Accuracy / 0.6231 NLL / 0.4331 Brier / 0.0283 ECE-15; audio+question
scored 66.96% / 0.6247 / 0.4340 / 0.0255. Paired uploader bootstrap for the
accuracy delta included zero. These are development results, not publisher
validation results.

No publisher-validation CSV, labels, or media have been downloaded or opened
or media have been opened for scoring as of this protocol revision. The
validation CSV was fetched after the initial protocol was committed, with its
publisher MD5 verified; only its schema and first row were inspected to confirm
field names. Before aggregate counts, cohort inclusion, feature extraction, or
scoring, the normalized-question rule above was corrected to match the prior
Clotho split audit, which explicitly treats generic wording shared across
different audio sources as non-overlap. This correction is not based on a
performance result. No model or GPU job is started by registering the protocol.
The run must not proceed unless the exact rights, identity, and minimum-group
gates above pass.

# TallyQA option-policy integrity correction — 2026-11-12

## Decision

The previous four-choice TallyQA development result is retained, but it is not
evidence of image-counting quality. Its option construction leaks substantial
label information through the **unordered candidate set**. Deterministic option
shuffling does not remove that information.

The source rows, old task manifest, image feature cache, readout checkpoints,
predictions, and old run directory remain preserved. A separate 16-choice task
manifest was generated with every count from 0 to 15 supplied for every row;
candidate order depends only on the fixed seed and question ID. The target is
bound after candidate ordering. This fixes the candidate-set shortcut, but its
existing validation rows were previously used by the four-choice development
probe. They are **not blind** and were not scored again here. Instead, a
separate inner-dev holdout was drawn only from old training rows and used for
one exploratory 16-choice readout comparison; results remain development-only.

## Confirmed shortcut

The old policy selected the answer plus nearby counts. For answer values 3
through 13, the unordered option set uniquely identifies the answer. For
example, `{1,2,3,4}` can only be produced by answer 3 and `{11,12,13,14}` only
by answer 13. Thus 11 of the 16 possible labels are recoverable from the option
set alone. On the preserved 3,101 training rows, there are 13 distinct option
sets; 368 rows occur in sets with only one observed training label. A lookup
that memorizes the most common label per option set gets 1,722/3,101 = 55.53%
on those same training rows. This is an **in-sample shortcut diagnostic**, not
a held-out score.

The old report's 55.93% joint and 56.30% question-only validation results are
retained as historical measurements, but they cannot be interpreted as clean
counting accuracy under this construction. The image-zero and shuffled-image
controls still show sensitivity to image features; they do not remove the
candidate-label shortcut or establish robust visual counting.

## Corrected task representation

- New task ID: `tallyqa_vg_amt_16_choice_label_independent_counting`.
- Every row has the same set of 16 integer counts, `[0, 15]`.
- A SHA-256 ordering of seed 17 and question ID permutes the options without
  reading the answer. The target index is computed only after that ordering.
- Existing image/question grouped split and media receipts are retained and
  checked for cross-split image, normalized-question, and media-hash overlap.
- Source revision, archive, source-screen, and acquisition hashes are pinned by
  the new generator. Only the `train.json` member is read; the official test
  member and sealed audit are not opened.
- The existing frozen image feature cache was reused read-only. A seed-17
  inner development split was formed only from the 3,101 legacy train rows:
  2,478 train / 623 dev questions, 2,236 / 559 images, with zero image,
  normalized-question, or media-hash overlap. It was evaluated once using the
  fixed 16-choice task. Because every row comes from a pool already used in the
  old four-choice fit, these results are development-only and not blind.
- On the 623 inner-dev rows, the image+question readout reached 52.65% accuracy,
  NLL 1.313, Brier 0.625, ECE 0.051. The question-only control reached 52.97%,
  NLL 1.344, Brier 0.633, ECE 0.039. Zeroing images yielded 46.87% / 1.432 /
  0.672 / 0.080; shuffling images yielded 47.99% / 1.426 / 0.679 / 0.081.
  Image inputs affect predictions, but joint accuracy did not exceed question
  only. Image-cluster paired bootstrap (2,000 resamples, seed 17) gave an
  accuracy difference interval of [-3.06, 2.40] percentage points for
  question-only minus joint; this is descriptive only because dev is drawn from
  prior training data and used for checkpoint selection.
- Each training run used 2,478 or 623 rows, 16 epochs, and one seed. Best epoch
  was 14 for both readouts. This small readout fit took 4.43 seconds (joint)
  and 3.67 seconds (question-only); peak GPU allocation was 28,847,616 bytes.
  The full pre-quantization resident count is 131,951,873 parameters
  (VideoPrism + MiniLM + readout), below the provisional 200M target.
- The 801 prior-exposed validation rows and official test were not loaded in
  this fit. Sealed audit was not loaded. No blind or product-quality claim
  follows from this diagnostic.

## Reproduction and artifacts

The old manifest SHA-256 is
`59c1c03edf17a8da6c870d4827d033af58759c8300f953b60cb0abbff98f8a45`. The
training-only audit was written to
`D:/CodexArtifacts/pretrained-reuse-r2/tallyqa-vg-label-independent-20261112-attempt-01/candidate-set-leak-audit-attempt-02.json`
(SHA-256
`326a8fc755255faecc2ca3979212a43e383e331d4a4c3efd111c5f458e4a7174`).

The initial full 16-choice task manifest is at
`D:/CodexArtifacts/pretrained-reuse-r2/tallyqa-vg-label-independent-20261112-attempt-01/task-manifest.json`
(7,607,653 bytes; SHA-256
`13561149f377a7b831a34516a0ac55d0569af23f856caa2a7e1fc6b47ebb2d8b`). It has
3,101 train and 801 previously observed validation rows. For the separate
development-only inner split and fit, use
`D:/CodexArtifacts/pretrained-reuse-r2/tallyqa-vg-label-independent-20261112-attempt-03/inner-task-manifest.json`
and
`D:/CodexArtifacts/pretrained-reuse-r2/tallyqa-vg-label-independent-20261112-attempt-03/readout/result.json`.

The source-screen document's previously mistyped screen-file SHA-256 was
corrected to the actual file hash already recorded in the machine-readable
manifest: `110fbbe4983106dceb90e82d5454aa40dd3eaebd6151fae03c8f6e6af40a516a`.

## Verification and next gate

CPU regression coverage verifies that the old policy exposes labels, the new
candidate set is independent of the answer, targets bind correctly after
ordering, 16-option readout encoding/training/metrics work, grouped inner
splits and cache subsets preserve disjoint identities, and split overlaps fail
closed. Ruff passes; 21 focused tests pass.

Do not train or select from the prior-exposed validation split. A future
TallyQA model comparison needs a genuinely independent, group-disjoint task
split or another admitted source. This integrity correction does not close the
product image-quality gate, the joint-multimodal gate, or the broader R2 gate.

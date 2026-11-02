# AVATAR Multi-Entity cross-source choice corpus freeze (2026-11-02)

## Status

Attempt-03 was invalidated by a post-freeze serialization-order shortcut.
Attempt-04 fixes that target schedule but remains a metadata-only diagnostic,
**not admitted for fitting**: its validation set has 68 examples in one
query/candidate dependency component, below the R0 evidence floor. Neither
attempt is model-quality evidence. No AVATAR media bytes were read by these
builders; no encoder, feature extraction, GPU job, or model fit was started.
Official validation and sealed audit remain unopened.

The task is a constructed four-choice cross-source category match: use a
2-second audio window as the query and choose the video-frame option from a
different source-video group carrying the same single AVATAR Multi-Entity
category. The task uses AVATAR TRAIN annotations only. It does not test
same-scene entity localization. The question is constant; scoring must use
only frozen audio/video tensors. Asset keys and timestamps are loader/alignment
metadata, and target labels, category labels, source groups, and sidecars are
not scorer features.

## Frozen corpus

Dataset revision is `mipal/AVATAR@a7f0146ff704799d044133e61137836cacc1c233`.
The 19,159-file TRAIN annotation tree hash is
`4c17a0bdbd7d488bced018404ce71be5659d27273efcfdec9e1cdfffd16eb651`.
Prior fit/evaluation source groups were removed using the inventoried usage
artifacts. The remaining 452 source groups were assigned by the seed-17
SHA-256 rank split (80/20), yielding 362 train and 90 development-validation
groups, with zero source-group overlap.

| Split | Groups in split | Single-category groups | Queries | Categories | Target positions |
|---|---:|---:|---:|---:|---|
| Train | 362 | 350 | 341 | 53 | 86 / 85 / 85 / 85 |
| Validation | 90 | 87 | 68 | 22 | 17 / 17 / 17 / 17 |

Groups that could not form an example were retained as documented exclusions:
12 ambiguous and 9 without a same-category positive in train; 3 ambiguous and
19 without a same-category positive in validation. No group lacked three
negative categories. Candidate and query source groups, and their asset keys,
are disjoint across the two generated folds. Perceptual media duplicates have
not been checked.

An option-position-only baseline selected from train predicts option 0. Its
train accuracy is 25.22%; validation accuracy is 25.00%, equal to four-choice
chance. No feature model was fitted for this control. The run artifacts do
not expose labels, source groups, or target positions as model inputs.

The corrected metadata files are external run artifacts at
`C:\CodexArtifacts\pretrained-reuse-r2\avatar-multientity-choice-corpus-20261102-attempt-04`.
Their hashes and the corpus manifest hash are in the attempt-04 machine-readable
record. Attempts 02 and 03 remain preserved. Attempt-03's four input and
sidecar hashes matched attempt-02, but its label schedule is invalid as
described below. Attempt-04 retains the same query/group selection and changes
the position schedule; its files have distinct hashes.

## Reproduction and integrity

Attempt-03's command is retained in the previous commit. The corrected
attempt-04 was generated with:

```powershell
python -m scripts.build_avatar_multientity_choice_corpus `
  --train-annotations C:\CodexArtifacts\pretrained-reuse-r2\avatar-av-decision-20261101-attempt-01\train-validation-annotations-02\train `
  --artifact-root C:\CodexArtifacts\pretrained-reuse-r2 `
  --output-dir C:\CodexArtifacts\pretrained-reuse-r2\avatar-multientity-choice-corpus-20261102-attempt-04
```

It reads TRAIN annotations and reconciles prior non-audit fit/evaluation
sidecars. Missing inventoried sidecars now fail closed. The attempt-04 script
SHA-256 and source commit are recorded in the companion repository manifest.
CPU tests:
`python -m pytest tests/test_audit_avatar_multientity_exposure.py tests/test_build_avatar_multientity_choice_corpus.py -q`
reported 9 passed; focused Ruff reported all checks passed.

## Post-freeze integrity correction

The first position baseline checked only marginal target counts. In
attempt-03, `target_index = len(inputs) % 4` made serialized row index modulo
four predict every target: **100% train and 100% validation**. Attempt-03 is
preserved, but must not be used for fitting, checkpoint selection, or model
evaluation. The original 25% constant-position baseline did not detect this
sequence shortcut.

Attempt-04 assigns a balanced schedule after a seed-17 shuffle independent of
serialized query order. The row-index-modulo-four diagnostic is 22.87% train
and 25.00% validation; the constant option-position baseline remains 25.00%
on validation. No model was fit. The attempt-04 manifest records exact file
hashes, target schedule, and the row-order control.

The validation query/candidate graph has 68 query groups and 87 total source
groups, but **one connected dependency component**; a candidate source group
is reused up to 12 times. Treating the 68 query groups as independent bootstrap
units would ignore shared candidate dependence. R0 requires at least 1,000
examples and 200 independent source groups for a quality PASS. This candidate
fails that evidence floor; do not spend storage on its media or claim that the
position fix makes it a sufficient benchmark.

## Limits and next gate

This set has only 68 validation queries and 22 represented categories. The
upstream AVATAR TRAIN annotations were already extracted and inspected, so
“prior-unexposed” means absent from inventoried model-use groups, not blind
labels. The AVATAR dataset declaration is CC BY 4.0, but AVATAR clips originate
from VGGSound/YouTube and original video rights remain with their owners; do not
redistribute media. Exact VideoPrism pretraining overlap with these sources is
unknown (see the R1 provenance audit). The task remains a narrow, derived
category match, not native AV localization or general multimodal decision
quality.

No fitting or media preflight should be started on this corpus: its sample and
dependency support cannot meet the current quality-evidence floor. A future
AVATAR experiment needs a redesigned, prospective task/split with enough
independent source support before media access. At the latest check C: had
about 1.26 GiB free, below the 2 GiB reserve. Existing attempt-02/03 outputs,
source archives, and model artifacts were not changed; no product promotion
is justified.

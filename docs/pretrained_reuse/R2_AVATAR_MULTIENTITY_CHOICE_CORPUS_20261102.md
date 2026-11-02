# AVATAR Multi-Entity cross-source choice corpus freeze (2026-11-02)

## Status

`METADATA_FROZEN_PENDING_MEDIA_PREFLIGHT`. This freezes a small development
corpus definition only. It is not source admission, model training, a quality
result, native AVATAR localization, or product evidence. No AVATAR media bytes
were read by this builder; no encoder, feature extraction, GPU job, or model
fit was started. The official validation and sealed audit remain unopened.

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

The frozen files are external run artifacts at
`C:\CodexArtifacts\pretrained-reuse-r2\avatar-multientity-choice-corpus-20261102-attempt-03`.
Their hashes and the corpus manifest hash are in the companion machine-readable
record. Attempt-02 was preserved. All four model-input/analysis-sidecar hashes
are identical between attempts 02 and 03; attempt-03 adds explicit feature
contract and metadata-only position-prior fields to the manifest.

## Reproduction and integrity

The builder command was:

```powershell
python -m scripts.build_avatar_multientity_choice_corpus `
  --train-annotations C:\CodexArtifacts\pretrained-reuse-r2\avatar-av-decision-20261101-attempt-01\train-validation-annotations-02\train `
  --artifact-root C:\CodexArtifacts\pretrained-reuse-r2 `
  --output-dir C:\CodexArtifacts\pretrained-reuse-r2\avatar-multientity-choice-corpus-20261102-attempt-03
```

It reads TRAIN annotations and reconciles prior non-audit fit/evaluation
sidecars. It refuses an existing output path. The script SHA-256 and source
commit are recorded in the manifest. CPU tests:
`python -m pytest tests/test_audit_avatar_multientity_exposure.py tests/test_build_avatar_multientity_choice_corpus.py -q`
reported 8 passed; focused Ruff reported all checks passed.

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

Before any fitting, a separate media preflight must verify selected audio/video
stream availability, timestamp alignment, decode integrity, and the permitted
local use of the source clips. Then establish fixed audio-only, video-only,
joint, missing-stream, shuffled-stream, and metadata-only controls on this same
group split. Keep this candidate unadmitted if those gates or a useful
independent evaluation support floor cannot be met. At the recorded live
check, C: had about 1.37 GiB free, below the 2 GiB reserve; therefore no media
or features were extracted. Existing artifacts were not changed, and no
product promotion is justified.

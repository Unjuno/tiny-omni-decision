# AVATAR Multi-Entity exposure and feasibility audit (2026-11-02)

## Result

This is a metadata-only eligibility screen for a possible native audiovisual
Multi-Entity task. It is not a model evaluation, task admission, or evidence of
quality. No model was loaded or trained, no media bytes were read, no official
validation annotations were opened, and no sealed-audit data was accessed.

The pinned AVATAR TRAIN annotation tree contains 3,286 source-video groups;
565 contain Multi-Entity annotations. I reconciled the query and candidate
source groups appearing in the saved AVATAR fit/evaluation artifacts: the
legacy option-choice sets, corrected cross-source readout, held-category fit,
inner validation and predictions, DINOv2 fold, and native-localization fold.
Across those separate experiments, the aggregate contains 2,271 groups in fit
artifacts, 1,681 in evaluation artifacts, and 2,576 in their union. The
1,376-group fit/evaluation intersection is **across experiments** and must not
be interpreted as a within-run split-leakage count.

Of the 565 Multi-Entity groups, 113 occur in at least one inventoried prior
fit/evaluation artifact (65 occur in both fit and evaluation artifacts). The
remaining **452 source groups** are absent from those prior model-use sets.
This is a model-exposure statement, not a claim that their labels were blind:
all TRAIN annotations had already been extracted and preflighted. The
source-group set is hashed in the machine-readable receipt; no group IDs are
emitted.

The 452 groups contain 523 annotated clips and 2,517 Multi-Entity annotation
rows with four-value bounding boxes. They cover 65 categories, but only 39
categories have at least 20 annotation rows and the least-supported category
appears in one source group. Only 5 annotated frames have two distinct
categories; 19 clips across 15 groups contain multiple categories. This
distribution may constrain a native within-scene entity-choice task. It does
not rule out a different AV matching task, which would need its own frozen
construction and controls.

A label-independent feasibility check applied a deterministic SHA-256 rank
with seed 17 and an illustrative 80/20 source-group partition. It produced
362 train / 90 validation groups (zero group overlap), with 416 / 107 annotated
clips. Forty-one categories occur in both folds, but only 5 categories have at
least four source groups in each fold; the weakest shared category has one
validation group. This partition is **not frozen**, and no model or score was
produced. The support counts argue against treating it as a broad category
benchmark without a task-specific design and a predeclared support policy.

The existing extracted-media directory contains 3,009 files, but none has a
source-group filename matching the 452 candidates. The already-downloaded
AVATAR video archive is present, but its contents were not read in this audit.
The free space snapshot was 1.38 GiB, below the roadmap's 2 GiB reserve. No
media extraction, feature generation, GPU work, or fitting was started.

## Reproduction

Source checkout at audit start: `788ee1e55f4865688de1678a038eebfac7406d59`.
Dataset revision: `a7f0146ff704799d044133e61137836cacc1c233`. The TRAIN
annotation tree hash is `4c17a0bdbd7d488bced018404ce71be5659d27273efcfdec9e1cdfffd16eb651`;
its 19,159 JSON files came from the repository's pinned AVATAR metadata archive
(`97e9f09e9f4325ac888db2886da22ca4650eafb8848721230a4e5d80fb6a4183`).

The audit command was:

```powershell
python scripts/audit_avatar_multientity_exposure.py `
  --train-annotations C:\CodexArtifacts\pretrained-reuse-r2\avatar-av-decision-20261101-attempt-01\train-validation-annotations-02\train `
  --artifact-root C:\CodexArtifacts\pretrained-reuse-r2 `
  --extracted-media-dir C:\CodexArtifacts\pretrained-reuse-r2\avatar-av-decision-20261101-attempt-03\train-validation-clips `
  --output C:\CodexArtifacts\pretrained-reuse-r2\avatar-multientity-exposure-20261102-attempt-04\result.json
```

The command refuses to overwrite an existing output and requires the
annotation directory to be named `train`. Final result SHA-256:
`f2bcd8f791462bdbfcd1beaa8252a536a8097683f9f3b73cc948685fff92149b`.
Audit script SHA-256:
`50a28d8e9d4ddabbe886df2983b13c674bf1c48c1000d069000e7f59c0badd4d`.
The result contains the hashes of every input sidecar/manifest and hashes of
the aggregate source-group sets. It contains no source-group IDs.

Verification: `python -m pytest -q tests/test_audit_avatar_multientity_exposure.py`
— 5 passed; Ruff passed for the new script and tests. The complete test suite
was not run in this metadata-only increment.

## Next gate

The 452 TRAIN groups are a possible development pool, not a sealed audit. Before
fitting, define a task that genuinely requires audio and video, freeze a
source-group-disjoint train/validation split, verify candidate and query
identities do not cross it, and establish media/audio stream availability for
the selected groups. The source archive is already local, but the current disk
reserve is insufficient for a new media-processing run. Do not create features
or train until the reserve and media-access path are verified. Keep the
existing AVATAR source classification and all prior artifacts unchanged.

# Perception Test: train-only task feasibility audit

## Decision

Retain as a concrete next paired-task candidate, **REVIEW**, not admitted to
training. Existing CREMA-D screening already excludes that narrow acted-affect
task; do not repeat its acquisition or substitute emotion recognition for the
requested general decisions.

Perception Test has same-video sound intervals, source object IDs and source
visibility labels. A prospective decision task is matching a heard event to its
visible source, with explicit invisible/unknown semantics. This is a candidate
requiring label/synchronization review, not an implemented benchmark. Known
source IDs are supervision only, never encoder input. Empty/unknown parents must
not silently become negative examples. All valid source candidates must remain;
do not select easy clips or discard invisible events to improve accuracy.

The existing multiple-choice questions cannot simply be called AV questions:
the train skill tags have no explicit audio/sound category. Having a soundtrack
does not demonstrate that both modalities are necessary.

## Pinned official evidence

Repository revision `3938d2f1ba3a6b502025741cea4cd73c7b3bdfaf`.
[Official README at that revision](https://github.com/google-deepmind/perception_test/blob/3938d2f1ba3a6b502025741cea4cd73c7b3bdfaf/README.md)
states Apache-2.0 for software and CC-BY-4.0 for other distributed materials.
Preserve attribution, source hashes and upstream citation when making a manifest.
This rights statement does not resolve the task and grouping gates below.

The official train annotation archive was downloaded without media or validation
and test annotations. Its SHA-256 is
`9eaa6aad3fabdf4cde4955f333cfddf3b9e6839c5abcb271afa410870f640d78`.
It contains only `all_train.json` (132,417,530 uncompressed bytes).
URLs for storage objects are mutable; the archive/cut-map byte hashes bind this
audit in addition to the Git revision. Do not call a storage URL immutable.

## Actual train-only counts

| Item | Count |
|---|---:|
| Videos | 2,184 |
| Three-option questions | 7,392 |
| Videos with sound annotations | 2,149 |
| Sound segments | 25,757 |
| Segments whose parent IDs all resolve to annotated objects | 17,558 |
| Videos with at least one such segment | 2,048 |
| Visible sound sources | 19,170 |
| Invisible sound sources | 6,587 |
| Videos requiring upstream answer-reveal truncation | 135 |

Visibility and resolvable-parent counts are separate properties. No missing cut
map entries, invalid positive cut frames or nonpositive sound intervals were
found by this bounded check. This is not full annotation or media validation.

## Required next gates

1. Recover contributor/recording-environment grouping. The inspected metadata
   exposes video IDs and technical fields, not contributor IDs. Unique video IDs
   alone do not establish household/actor-disjoint train/validation/audit.
2. Define event/source option semantics and invisible/unknown handling before
   scoring. Verify which labels genuinely need audio plus video rather than a
   single-modality shortcut; include fixed unimodal and shuffled-source controls.
3. Honor upstream cut-frame mappings: train/validation videos can reveal answers
   in their endings. Trim both audio and video consistently before extraction.
4. Verify bounded media acquisition with checksums and enough disk. Official
   whole-train video archive is listed as 26.5 GB; do not download it blindly.
5. Freeze grouped splits and task protocol before validation inspection, with
   at least 200 independent scenes when feasible and grouped uncertainty.

## Reproduction and artifacts

`python -m scripts.audit_perception_train --source C:/CodexArtifacts/pretrained-reuse-r2/perception-test-source-audit-01 --output <fresh-result.json>`

Local directory contains the pinned README, train annotation archive, train
cut map and result.json with all hashes, tags and sound classes. The audit checks
train identities, three-option targets, cut ranges and source-parent references.
Unit testing rejects non-train input. No model inference, fitting, media download,
validation/test access, user form submission or cloud cost occurred.

## Follow-up: official grouping evidence

The [paper's Appendix 5](https://arxiv.org/html/2305.13786#A5) specifies grouping
by the combination of filming script and participant. Alternate viewpoints of
the same performance remain together. It does not promise participant-disjoint
splits: the same participant could contribute different scripts to different
splits. The appendix also explains that question templates recur across splits.

Inspection of the complete repository tree at the pinned revision found README,
license/contribution files, visualization and baseline notebooks, but no
participant-to-video mapping. The inspected train metadata likewise contains no
participant or script IDs. This is scoped evidence about these inspected files,
not proof that such a mapping cannot be obtained elsewhere.

Therefore preserve official splits for any prospective development experiment;
do not randomly subdivide videos from official train and call them independent
validation. Official grouping is useful provenance, but does not meet a stronger
participant/household-disjoint gate. Do not rename official validation as a sealed
audit. A stricter split needs an authoritative mapping; do not infer identities
from faces/voices or guess groups from adjacent numeric video IDs.

No external message requesting metadata was sent. No training or validation
acquisition was initiated. Next independent work can define the train-only task
adapter and truncation contract, while keeping this generalization limit open.
The paper HTML and pinned repository-tree response are preserved alongside the
source audit. This follow-up resolves what upstream guarantees; it does not
resolve the missing fine-grained grouping information.

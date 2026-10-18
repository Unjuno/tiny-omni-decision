# Perception occlusion: source roles and temporal cues

## Executed result and scope

Eight previously selected, known-source **training** clips were checked against
the pinned original annotation archive. All eight have distinct grounded-QA
answers for the hidden item, its covering object and the similar-object group.
The covering IDs agree with the unchanged earlier TAPIR/SAM2 diagnostic plan.
This resolves the source role association; **input-only target identification,
game-time anchors and actor-relative answer mapping remain unmeasured**.

The CPU audit decoded and visually inspected eight uniformly spaced raw frames
per clip,64 frames total. It took26.017s for frame extraction/contact-sheet
creation. No model inference, training, validation, sealed audit or new download
was performed. These are sparse, label-aware observations on a reused training
subset, not a new blind benchmark or dense event annotation.

## Source evidence

Perception Test source revision
`3938d2f1ba3a6b502025741cea4cd73c7b3bdfaf` documents separate
[object tracks, grounded answers and action segments](https://github.com/google-deepmind/perception_test/blob/3938d2f1ba3a6b502025741cea4cd73c7b3bdfaf/README.md).
Grounded answers refer to track IDs; multiple-choice answers refer to option IDs.
These namespaces must remain separate. The local training archive's SHA-256 is
`9eaa6aad3fabdf4cde4955f333cfddf3b9e6839c5abcb271afa410870f640d78`.
Its sole member is `all_train.json`; only the fixed eight rows were inspected.

| Train video | Hidden ID / source label | Cover ID | Similar IDs | First hidden annotation | Source shuffle segment |
| --- | --- | ---: | --- | ---: | --- |
| 10103 | 6 / apple | 4 | 4,5 | 0 | 289–424 |
| 10186 | 6 / eraser | 4 | 3,4,5 | 0 | 407–612 |
| 10212 | 5 / cap | 2 | 1,2,3 | 0 | 286–894 |
| 10387 | 2 / cap | 10 | 9,10,11,12 | 0 | 290–491 |
| 1087 | 7 / purple-toy | 2 | 2,3,4 | 0 | none annotated |
| 10886 | 3 / box | 0 | 0,1,2 | 0 | 280–621 |
| 10894 | 4 / vaseline-bottle | 1 | 7,1,2 | 0 | 194–534 |
| 10903 | 7 / cap | 3 | 1,2,3 | 224 | 298–525 |

IDs, labels and event segments above are **diagnostic/training supervision**, not
deployable inputs, predictions or automatically detected events. The similar
groups contain two objects on one clip, three on six, and four on one; each
agrees with its upstream counting answer. Three supplied answer options do not
mean there are always three physical covering objects. No question was removed
or option count reduced. Missing shuffle annotation on1087 does not imply that
the clip is broken: its action list includes movement of another object.

## What the raw frames establish

- Frame0 commonly shows objects already covered. Some prelude frames show an
  object during lifting, for example the small purple item in10894 frame78 and
  the cap in10903 frame239. Correctly associating a reveal with a covering object
  requires temporal observations; these observations do not test an automatic
  association algorithm.
- The hidden-item track in10903 starts at224, within its source annotated lift
  segment214–289. A first-frame mask probe cannot measure discovery of that
  later-revealed item. Source annotation availability is not a model prediction
  or a complete visibility label.
- Some clips show an actor across the table; others show a hand from approximately
  the camera side or an actor at the table's side. A universal reversal of
  image-space left/right is not justified.10903 uses closest/farthest options,
  and10387 has four physical covering objects. Keep actual option text and
  viewpoint/depth semantics rather than mapping answer indices to a fixed rank.
- Source `is_masked=0` alone is not a verified target-visible cue. No mask flags,
  boxes, grounded IDs, answers or action intervals chose the reviewed frames.
  The full visibility/event semantics remain UNKNOWN for a deployable selector.

Eight frame indices per video were fixed by rounded linspace over the official
input interval. For a nonnegative source cut, all indices are strictly before
that cut.10212 has official cut=-1; its full official input was retained. Later
lifting occurs in that clip; this audit neither invents a cut nor establishes
that a particular frame reveals a final answer. Answer-reveal policy requires
separate source-level confirmation before declaring a new quality evaluation.

## Consequence for the next experiment

The previous [SAM2 initialization result](R1_SAM2_AUTO_INIT_20261014.md) stays
unchanged:8/8 raw/NMS initial covering-box opportunity,2/8 under fixed top3
presence ranking. These counts are not hidden-item discovery or QA Accuracy.
Source roles are now known for this subset; the model still has no measured
input-only way to recover the relevant association.

Before a query-dependent fit, freeze a bounded protocol that observes the
prelude/history, keeps variable-sized object groups, and specifies actor-relative
geometry and explicit unknown/ambiguous cases. Grounded IDs/boxes/action labels
may provide training targets only. Frame selection and inference must use raw
inputs and the official cut, with no answer-conditioned initialization, GT-match
seeding or target-dependent threshold search. Measure target association and
temporal retention separately before attributing a Decision improvement to them.
More epochs on the existing first-frame substitution are not justified by this
audit. General95–98% Omni quality and genuinely useful joint inputs remain open.

## Artifacts, reproducibility and failures

Source commit `f5bb5241097b338f4770b72f3df18c5b851646cc`.
Local root `C:/CodexArtifacts/pretrained-reuse-r2/perception-source-cues-01/`:

- `annotation-inventory.json`: full eight-row role/action/question inventory.
- `raw-frame-review-02/result.json` and eight PNG contact sheets: media hashes,
  exact frame indices and image hashes. Images contain raw input frames and
  frame-number captions only, with no annotation overlays.
- `result.json`: independently checkable role table, limits and artifact hashes;
  identical semantic content, with LF publication bytes, in the adjacent
  [JSON](R2_PERCEPTION_SOURCE_CUES_20261014.json). Original Windows bytes retained.
- `verification.json`:16 saved-artifact/image hashes independently rechecked,
  table counts recomputed, prior SAM2 result/checkpoint hashes unchanged.
- `frame-attempt-01-failure.json`: first plot attempt failed before decoding
  because matplotlib is absent. Its runner remains preserved. A separate attempt
  used existing Pillow; no dependency upgrade or data-policy change.

External scripts `perception_source_cues_01.py`, `perception_cue_frames_01.py`,
`perception_cue_frames_02.py`, `perception_cue_report_01.py` are preserved at the
parent artifact root with hashes in the report. Python3.11.9, existing PyAV/Pillow/
NumPy, CPU only; GPU time0, cloud cost0. Original local report SHA-256
`60446f67a2bc092a427db789ff245644255fd8f4cf0f0910612a9bcd184bf905`.
LF publication SHA-256
`e1531811c16e0d6fe2cdc728070b503f1d0969062b47a0e0d03431368cf0676c`.
The old SAM2 result/checkpoint SHA values were rechecked unchanged. No product
code, weights, media, legacy configuration or checkpoint was modified.
Local verification: Ruff passed,953 CPU tests passed and1 skipped in15.61s;
model/dataset manifests, four catalogs and the example policy audit passed.

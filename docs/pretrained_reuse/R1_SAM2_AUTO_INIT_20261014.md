# SAM2 geometric-grid initialization: executed, target selection unresolved

## Measured result

The full fixed eight-training-video probe completed with exit0:
`PASS_AUTO_INIT_EXECUTION_ONLY`. All8,192 first-frame proposals were generated
without annotation prompts and retained before scoring. NMS survivors contain a
box overlapping the annotated **covering-object/occluder track** at IoU>=0.5 on
8/8 clips; fixed top3 presence-score selections contain one on **2/8**.

This is initial covering-object **bounding-box opportunity**, not pixel-mask IoU,
hidden-item identity, correct semantic target selection, tracking Accuracy,
Decision Accuracy or95–98% Omni quality. Best-overlap indices are post-hoc oracle
matches for diagnosis; they were never used to choose or seed a track. Only the
model-score-selected top3 were initialized and propagated one adjacent frame.

## Fixed probe and input isolation

- Same published SAM2.1 Tiny checkpoint/revision and existing Transformers5.6.2
  backend as [R1 admission](R1_SAM2_TINY_20261014.md), all38,962,498 parameters
  counted and frozen; FP32/SDPA, seed17, TF32 off, deterministic, no compilation.
- Static IDs of the same first eight eligible train clips used by the previous
  [TAPIR train diagnostic](R2_TAPIR_EXPANDED_PROTOCOL_20261013.md). Previous
  eligibility required a unique occluder track, sufficient unmasked annotations
  and an annotation at frame0. That is a limited, previously inspected train
  subset, not a representative test set or a new blind audit. No clips replaced.
- Inference reads a sanitized media manifest containing only video ID/path/SHA,
  train split, reveal cut and frame indices[0,1]. No annotation plan, target box,
  track ID, answer, text query or object class is loaded by the inference script.
  File/source names are output bookkeeping, never neural features.
- Published Perception Test source3938d2f1ba3a6b502025741cea4cd73c7b3bdfaf declares
  CC-BY4 for distributed materials. Existing cached media/checksums are reused;
  no new media download. Broader participant grouping/paired-task admission
  remains unresolved, as documented by the [source audit](R2_PERCEPTION_TEST_SOURCE_20261012.md).
- RGB frame0, fixed32×32 cell-center grid,64 distinct object IDs per chunk.
  One generated positive point per object; no ground-truth location supplied.
  Eight clips×1,024 point/mask proposals, all saved. The32-side grid follows the
  existing mask-generation pipeline's default density, but this wrapper uses the
  complete VideoModel API and is **not the official automatic-mask generator**.
- Pixel foreground is raw logit>0. Model **object-presence** logit ranks candidates;
  stable ties use row-major grid index. Empty masks cannot seed a track; all
  empty/nonempty outputs remain recorded. Box NMS IoU0.7, then at most3 survivors
  selected, fixed before the model results. No target-dependent filter or threshold
  search. No predicted mask-IoU quality filter or semantic relevance score exists.
- Fresh video sessions receive only those model-produced binary masks. Two
  original adjacent frames[0,1] pass through the same processor. This tests
  initialization and plumbing, not dense long-video temporal identity retention.
- Preflight:12GiB free GPU;400s wall cap/6GiB Torch allocation cap. No training,
  optimizer, seed search, validation, sealed audit, adaptation or quantization.

Source-config SHA`371135a2b4ea735f65038312927a02f29541739dee2c7889a738d5c071578dd3`.
Failed and corrected attempts have **identical** effective-config hashes.

## Retained candidates and selected opportunity

Independent CPU scoring started after the complete inference bundle had been
saved. It rehashed/unpacked all proposals, recomputed boxes from packed masks,
regenerated grid coordinates, repeated deterministic ranking/NMS and validated
all propagated arrays. Scores, selected indices and masks were not changed.

| Train clip | Raw masks | NMS survivors | Best raw bbox IoU | Best NMS bbox IoU | Best top3 bbox IoU |
| --- | ---: | ---: | ---: | ---: | ---: |
| video_10103 | 1024 | 55 | 0.895466 | 0.776106 | 0.037456 |
| video_10186 | 1024 | 186 | 0.974838 | 0.974838 | 0.000000 |
| video_10212 | 1024 | 43 | 0.887175 | 0.877635 | 0.877635 |
| video_10387 | 1024 | 168 | 0.963966 | 0.929403 | 0.012843 |
| video_1087 | 1024 | 67 | 0.946966 | 0.946966 | 0.946966 |
| video_10886 | 1024 | 64 | 0.915263 | 0.908112 | 0.307927 |
| video_10894 | 1024 | 73 | 0.966247 | 0.945728 | 0.292086 |
| video_10903 | 1024 | 40 | 0.969927 | 0.969927 | 0.027929 |

All masks were nonempty in this fixed probe. Geometry overlap can be scored for
one annotated covering object; other true objects are not negatives, so no
all-object detector precision or recall is inferred. No confidence interval or
population claim is made from this selected eight-clip training diagnostic.

The annotation source is the unchanged old plan SHA
`977cfcd414db9a34af1f88a9089e2a5e6f1eb65d7e89dc32c57ac7d0199e5b32`.
Its boxes are normalized[x1,y1,x2,y2], initial annotation frame0. The raw/NMS
overlap maximum is computed with ground truth **only for scoring**. The top3
overlap is also a coverage diagnostic, not proof of which selected track is the
semantic query target. These are not correct-answer counts.

## Scope corrections and task semantics

The annotated training plan and earlier TAPIR outcomes were known/inspected
before this probe. The scorer's historical field
`annotation_opened_only_after_complete_inference` refers to the **separate scoring
process**, not global investigator blindness. A preserved scope addendum makes
that explicit. No annotation information was passed into the inference/ranker.
The published [JSON](R1_SAM2_AUTO_INIT_20261014.json) retains original records and
both clarifications; no historical model output or numeric score is overwritten.

`is_occluder` identifies a covering object, not the hidden item. As the prior
[annotation admission](R2_OCCLUSION_TRACK_ADMISSION_20261013.md) records, the exact
hidden-question target mapping is unresolved. Actual questions ask for the hidden
object's location at the beginning/end of the **game**, from the person's viewpoint;
options include actor-relative ordinals or relative distance. Frame0 need not be
the semantic beginning-of-game anchor. Initial-box coverage cannot validate that
binding, hidden-item-to-covering-object association, viewpoint or depth semantics.
Earlier ordinal diagnostics also had a [withdrawn index-mapping error](R2_TRAJECTORY_DECISION_RESULTS_20261013.md).
Do not train against these unresolved substitutions as if they were QA truth.

The results localize one part of the pipeline: useful initial geometry can exist
among many proposals while unconditional presence ranking discards it. Presence
is not task relevance. The ranker never received a question/target-identifying
cue, so2/8 does not prove a learned semantic selector or model capacity failed.
It also does not prove that all needed visual/temporal information is available.

**Next gate:** specify which observation/history/criterion identifies the relevant
covering object and the game's temporal anchors; verify source semantics on
training data before designing a query-dependent selector/readout. If the input
does not uniquely identify the intended target, report the ambiguity. Do not
replace the frozen top3 with post-hoc GT matches, sweep thresholds/TopK to inflate
these counts, immediately enlarge the model, or start Decision training/quantization.
General multimodal quality and useful joint evidence remain unqualified.

## Execution, invariants and unsuccessful attempt

Actual source commit`f295cd89e7f2cddadeee9b61075466524f8da9f5`; executed external
runner`auto_init02.py`, SHA
`b9918adcdf5aceed646b6d404a611618fe3026a9edec822beea382dd37e943c0`.
Wall150.972049s(2.516min); model peak Torch allocation1,089,160,192bytes and
reservation1,207,959,552bytes. Existing RTX3080 Laptop, Torch2.6.0+cu124. This
excludes driver/CPU memory and is not end-to-end Decision latency. Cloud cost0.
First64-point chunk fresh-session masks and scores repeat bitwise; all proposals
and propagated masks are finite. Model persistent state before/after SHA:
`658e3aaff3d08b6cc8337a1f125e5ee55451ed3bfec38f9def601a14bb6912d8`.
Checkpoint SHA remains`48c14467e5cf9e51870511feb72c89688e82dd74523142c0538b663e193ac2a7`.

The first attempt`perception-sam2-auto-init-01/` exited1 after its first64 proposals:
the processor stored the caller's mutable object-ID list, and model.forward
removed IDs from that list in place. Repeat then received zero IDs. This caller
alias error was reproduced on the actual CPU processor/state interface; passing
a fresh list prevents it. The regression uses direct state.clear() as equivalent
cleanup, while the actual model removes each ID individually. No library upgrade,
model/settings/data change. Original failed runner/result remain preserved.
Only caller-copy/output-path changes produced attempt02, with the same config SHA.

Local paths under`C:/CodexArtifacts/pretrained-reuse-r2/`:

- `perception-sam2-auto-init-02/`: config/progress/result, all8 packed proposal
  banks with scores/boxes/points/NMS/top3, and all selected two-frame outputs.
- `perception-sam2-auto-init-score-01.json`: independent after-execution geometry
  verification/diagnosis, no model inference after its annotation load.
- `perception-sam2-auto-init-score-scope-addendum-01.json` and
  `perception-sam2-auto-init-annotation-role-addendum-01.json`: precise provenance
  and annotated-role limits, original score unchanged.
- `perception-sam2-auto-init-01/`: failed first attempt, retained.

Probe and scorer scripts are throwaway research artifacts under
`C:/CodexArtifacts/pretrained-reuse-r1/sam2.1-hiera-tiny-audit-01/`.
No product provider or model training was added. All legacy media, teachers,
configurations/checkpoints and the previous SAM2 admission remain unchanged.
Local verification:953 passed,1 skipped in14.81s; Ruff passed. GPU inference and
after-execution geometry checks were separate from normal CPU CI. Published
JSON retains its audited LF bytes; no model weights, masks or videos enter Git.

# R2 native corpus — independent saved-prefix audit, 2026-10-14

## Actual evidence

**Saved-prefix audit PASS; full corpus still incomplete.**
[Audit protocol](R2_NATIVE_DYNAMIC_CORPUS_AUDIT_PROTOCOL_20261014.md).
Audit source commit `5d0d172180c3d6d3679a06d27d55c22c35ac5589`; native extractor remains the unchanged
`566af2fafd5895711e2cf206615ff067014e8fa9` code/config under PID38972/exec22985.
The extraction process was confirmed live separately; this captured progress
report alone is not evidence of liveness or terminal completion.

| Audited item | Captured count |
|---|---:|
| Source media assets actually RGB-decoded |2|
| Saved frame records |59/256 planned|
| New native frame records |57|
| Reused initial-frame records |2|
| Raw candidates independently checked |60,416|
| Full NMS survivors |4,969|
| Train10103 frames |32/32|
| Train10186 frames |27/32|

Capture `2026-10-14T07:02:07.875602+00:00`; audit end `2026-10-14T07:02:21.922421+00:00`.
Every saved1,024-candidate mask bank was unpacked with native big bit order.
Normalized boxes reconstructed exactly from mask extent; native source-grid
point order reconstructed exactly from image dimensions; stable score/index
NMS0.7 and no-object presence semantics checked independently. Empty proposals
remain in the raw inventory. No extractor/legacy geometry functions are imported
by the new audit helper.4,969 survivors are not4,969 unique physical objects.

All file/config/source hashes and reused parent full five-array bytes match.
Actual source video was independently decoded: every captured frame's native
RGB SHA-256 and dimensions match the extractor's record. Pinned source hashes
include all fixed inputs, but only the two assets/59 frames above were RGB-audited
in this captured prefix; future rows and six other source RGB banks are untested.

CPU body **14.100195s** after imports. It overlaps native extraction;
no isolated throughput/latency/energy/peak-CPU-memory claim. No model/GPU/weights
loaded by the auditor; a checkpoint file hash is read without loading tensors.
No annotations, source-role/visibility/QA labels, dev/sealed audit or training.
No main code/config/data/checkpoint or process intervention; source hashes are
checked again at audit exit. Captured report bytes are preserved separately.

## Checks and limits

23 new CPU tests first failed for absent APIs, then passed: hand-checked packed
mask/box/grid/NMS/tie/score ordering, empty inventories, nonfinite/schema/presence
rejection, input nonmutation, and ordered-prefix/full-terminal status separation.
Optimized Python is rejected so verifier assertions cannot be disabled.
Local full suite **1,278 passed/1 skipped,18.63s**; Ruff, model/dataset manifests,
four catalogs and policy validation passed. Read-only code review found no
Critical/Important issues before actual snapshot execution. No GPU tests ran.

The auditor never treats a snapshot as complete, even if it happens to contain
all256 rows. Full admission separately requires actual terminal native PASS,
complete ordered coverage (39 reused/217 new), final model-state/flags/RNG
checks, the predefined fresh10103/time15 all-five-array replay, full source/RGB
and artifact hash checks. These have **not** been completed in this record.
No Decision Accuracy/NLL/Brier/ECE, semantic cue-to-cover/identity understanding,
unique objects or joint/OOD95–98% quality is established by this input audit.

## Artifacts and continuation

[Small publication JSON](R2_NATIVE_DYNAMIC_CORPUS_PREFIX_AUDIT_20261014.json),
SHA-256 `f9a3925e153a0b41eca02a23060acb7faaf24e81350f864d6d0c37cc17de5a26`.
Audit result SHA-256 `0642504ab0079e0f7226f97ef7071ec870fed327b76b474be8a643d41f4aedbc`.
Auditor script SHA-256 `f15c539c4b4225151ed5e3fd9290f4f5bc7f2b78931e49004fd92abfb2dd74f9`;
independent helper SHA-256 `d0ea7b2cafadf455a99180e37c423a4997dcced1a027887bb3811a3e502cb473`;
fixed main config SHA-256 `5d4417e13d8c2a4f27593d3149ac4167e7d8ee13ff9c87a076ff9780f60aa79f`.

Local `C:/CodexArtifacts/pretrained-reuse-r2/native-dynamic-corpus-prefix-audit-01/`
contains the immutable captured input report and audit result. Weights/media/NPZ
stay outside Git. No batched-head adoption or further width/nearest-distance
search. Follow the existing native process unchanged; after terminal completion,
execute the already fixed fresh replay and full CPU admission before post-hoc
semantic supervision. This prefix result does not finish the overall Omni goal.

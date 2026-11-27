# IntPhys temporal-input coverage audit protocol (2026-11-27)

## Question and scope

Does uniform sampling of more than the current eight RGB frames reduce exact
same-scene, opposite-label collisions in the already inspected IntPhys dev
archive? This is a **data identity audit only**. It does not run VideoPrism,
produce features or model predictions, fit weights, select a checkpoint,
calibrate a score, or access the hidden test.

The dev archive has already been used for a fixed model diagnostic. It is not a
fresh or blind evaluation. Results from this audit can only describe input
identity and inform the next experiment protocol; they cannot establish task
accuracy or serve as independent quality evidence.

## Frozen source and policies

- Source: official IntPhys `dev.tar.gz`, previously admitted under CC BY 4.0.
- Archive bytes: `3,203,229,667`.
- Published MD5: `6a72715007f2f3b0e9546bfa8d8fc39b`.
- Previously verified SHA-256: `441eebe09f6583a18dbcd9d2b6c6b5667c66bac99e3875622e0768dc22e16a0a`.
- Expected structure: 360 videos, 90 `(block, scene)` groups, four renditions
  per group, and a 2 possible / 2 impossible label balance.
- Candidate frame counts fixed before execution: **8, 16, 25, 32, 50, 100**.
- For each count, use 1-based, endpoint-inclusive, evenly spaced indices over
  frames 1–100 using `floor(i * 99 / (count - 1)) + 1`; count 1 is
  defined as frame 1. The current eight-frame policy must yield
  `[1, 15, 29, 43, 57, 71, 85, 100]`.
- Per-frame identity is SHA-256 of the exact pinned PNG payload. The ordered
  sequence identity is SHA-256 over the selected frame hashes in time order.
  Labels are not used to construct identities; only after every sequence hash
  is computed are labels joined to count opposite-label collisions within each
  four-video scene group.

## Measurements and interpretation

For every fixed count, record selected frame numbers, ordered identity hash,
all 90 scene groups, all 360 possible/impossible cross-label pairs, the number
of groups with any exact cross-label sampled-sequence collision, and the
number of exact colliding pairs. Do not remove examples or alter the current
split. Do not compare model accuracy, thresholds, NLL, Brier, or ECE.

PNG-byte identity is an exact-source proxy, not decoded/resized VideoPrism
tensor identity. An exact sequence collision proves those sampled source
frames cannot distinguish the labels; no collision does not prove the frames
contain sufficient task information. Any candidate frame policy chosen after
this audit requires a newly frozen, independent evaluation before a quality
claim. The already observed IntPhys dev set must not be reused as a blind test.

## Output and implementation

The audit implementation is `scripts/audit_intphys_temporal_sampling_coverage.py`.
CPU-only unit coverage is in `tests/test_intphys_temporal_sampling_coverage.py`.
The result JSON is stored outside Git under
`C:/CodexArtifacts/pretrained-reuse-r2-local/intphys-frame-coverage-20261127-attempt-01/`.

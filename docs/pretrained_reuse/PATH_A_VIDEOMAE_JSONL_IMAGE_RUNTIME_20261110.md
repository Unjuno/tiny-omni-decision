# Path A VideoMAE image JSONL runtime — 2026-11-10

## Result

The task-scoped static-image Decision route runs end to end through the Path A
JSONL CLI on the local RTX 3080 Laptop. For one official Clevr-4 **training**
image, its probabilities exactly matched the already-frozen VideoMAE feature
cache and readout (`max_probability_delta = 0.0`). This is runtime/parity
evidence, not a new quality evaluation. No validation or sealed-audit labels
were loaded, no training occurred, and no accuracy or calibration was measured.

## Implementation

- Added task ID `path-a-image-videomaev2-clevr4-100k-runtime-v0`, routed to a
  separate persistent PyTorch worker. It does not launch the VideoPrism worker
  whose TensorFlow dependency is blocked by host policy.
- The worker verifies the pinned VideoMAE source, checkpoint, scorer, and local
  media SHA-256; it returns scores for the caller-supplied subset/order of
  taxonomy choices.
- Reused one preprocessing/scorer implementation from both frozen-feature
  extraction/training and runtime code to reduce preprocessing drift.
- Static images are repeated across the encoder's 16-frame input. This path
  has no temporal evidence and must not be described as video understanding.
- Encoder has 22,153,286 parameters; the frozen readout has 52,682; total:
  22,205,968 parameters. Runtime weights total 44,547,594 bytes.

## Pinning and run evidence

- Source branch base before these uncommitted integration changes:
  `a987e7e5abed9009bc1fb2db810f11c261c31738` (itself includes the existing
  Path A VideoMAE probe commit `28ffb63d`).
- VideoMAE source SHA-256:
  `d132c97369f25207e380e78c59294fd00b93d2f6aa0fb5c4ded19eff26a5a3cb`.
- Encoder checkpoint SHA-256:
  `24fb71687fa3671b8387cadfbcbab0f72af695692e93cf1ecc82caa888626172`.
- Frozen readout SHA-256:
  `437e920e7cda02ff2e28f19c96433af04aab3d70494030b6e3278809aa1cc39e`.
- Clean Path A assembly manifest SHA-256:
  `79fbca05e6062b66fe6ffb8a831d6c635edffe361c76ea4ce1574f0736ae9519`.
- Input media: `CLEVR_new_000000`, train split, SHA-256
  `3d4c1e813a18e6fafde4707fb6947fa6f1dfacc1ee55d3e9bb2ba8459055e0ac`.
- Device/software: NVIDIA GeForce RTX 3080 Laptop GPU; Torch 2.6.0+cu124;
  CUDA 12.4; FP32.
- Expected and runtime probabilities, in supplied option order:
  `[1.671119207458105e-06, 0.9999983310699463]`.
- Model load: 2.8895 s; worker startup: 5.0097 s; inference: 0.2902 s;
  dispatcher request: 5.3187 s; outer command: 7.9633 s.
- CUDA allocation after request: 98,793,472 bytes; peak: 240,896,512 bytes.
- VideoPrism worker started: no. Training: no. Validation/audit access: no.

The entrypoint was `python scripts/smoke_path_a_videomae_jsonl_runtime.py`
with the frozen input, encoder, readout, assembly, and output arguments. The
exact CLI argument string and fully resolved package dependency directory
were not persisted separately; those fields are `UNKNOWN`, rather than
reconstructed from memory. The smoke script refuses to overwrite an output
directory and checks hashes before extracting one train image into that new
directory.

## Immutable attempt receipt and correction

The successful run is in
`C:/CodexArtifacts/pretrained-reuse-r2/path-a-videomae-jsonl-image-runtime-20261110-attempt-02/`.
Its `result.json` says `attempt-01` in the `experiment_id` field due to a stale
script constant, even though the output directory and distinct request are
attempt-02. The result JSON was preserved unchanged; its SHA-256 is
`14b5c1586d63657b15bd5a426bee545839d867ff839cc7d91df6c0fa013ce8df`.
`metadata-reconciliation.json` records the corrected directory-derived ID and
explicitly records `result_json_modified: false`. The source smoke script now
derives that field from the output directory name. The request and response
SHA-256 values are respectively
`c459770881f921b88fa4c97f8467eb7157d660ebb2abad06d1e75c68774afb4f` and
`6d8a284e47ad0270e9ac7e3396e55ae061ffb05b0525b66502ce5f2d265fa742`.
The runtime module and worker hashes at execution were respectively
`fcacb730e17254dc83a31b45a88d98e11e63a6e516e619f52c5943a5ada5760c` and
`0447619c87ea83ab92aebd27c20f99ed68ac884dda985dc3d477d76f86471c52`.
The smoke-runner hash at execution was
`fefa43f43c3121d06f49f39b6fa202408c5e6ba10d39c00c97c051796cffbc17`; the
current runner hash is
`651c3889726bf64b6a70943654268312ddc243caf4c3f8431d3f6b11c3637998` after a
metadata-only correction making future experiment IDs derive from their
output directory. The original result remains unchanged.

## Checks and limits

Focused CPU tests passed: `12 passed` across JSONL dispatch, VideoMAE runtime
contracts, and frozen-feature extraction. Ruff check/format and CLI help pass
for changed files (the existing `cli.py` formatting baseline was kept intact).
Full pytest collection fails before test execution because the existing
`tests/test_freeze_videomae_clevr4_inputs.py` imports the untracked/missing
`scripts.audit_clevr4_content_overlap` module. The GPU smoke above is a
one-example train-split integration parity check, not a benchmark.

The existing Clevr-4 frozen-feature experiment reported 80.01% macro accuracy
(count 43.43%, texture 79.04%, color 98.70%, shape 98.89%). Those historical
results are not reproduced or re-evaluated here and are restricted to the
synthetic static taxonomy task. Natural-image task quality, VideoMAE temporal
video quality, cross-modal complementarity, joint audio-video quality, package
size reduction, and release qualification remain open. Existing VideoPrism
Image/Video routes remain separate; this change does not bypass Windows code
integrity policy or replace their environment.

## Preserved artifacts

Teacher v1/v2 configs, corpora, checkpoints, model artifacts, prior Path A
assembly, the shared user worktree, and other worktrees were left unchanged.
No model weights, media, or large checkpoints were added to Git.

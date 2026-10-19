# HoloAssist text-free sensor comparison — progress, 2026-10-19

## Why a new attempt is required

The completed HoloAssist temporal readout is not a text-free baseline: the
video-only, audio-only, and joint arms all received the MiniLM embedding of the
task question. Its 207 official validation sessions were used for checkpoint
selection and comparison, so they are not reused here. The prior result remains
unchanged in
[`R2_HOLOASSIST_TEMPORAL_READOUT_RESULTS_20261019.md`](R2_HOLOASSIST_TEMPORAL_READOUT_RESULTS_20261019.md).

## New development corpus

A new immutable plan was created outside Git at
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-attempt-01\selection.json`.
It uses only the official train member of `data-splits-v1_2.zip`; the validation
and test member payloads were not read. It excludes the old 160 training
sessions and partitions all 1,306 remaining official-train sessions into 1,042
training / 264 development-validation sessions, stratified by task category
without reading correctness labels. All 20 task categories occur in both
splits; rare categories have only two validation sessions and require cautious
interpretation. Session, video, and window identities are disjoint by
construction. Participant disjointness remains unknown because participant
IDs are not exposed.

- Selection seed: `20261019`.
- Selection SHA-256: `66f578b89ddb1757477b6c75e31b9d3fb7320d6832ae3d23dcd72ad67be17498`.
- Pinned media index SHA-256: `ad5a3843e6592d4be8eab1b187f36ff69f4fab90a9be57c06be2c0ce5fd2f8dc`.
- Allow-listed pitch-shifted media required: 117,074,880,152 bytes (117.07 GB).
- Seed is for deterministic grouping only; no model fitting or checkpoint selection has used this split yet.
- The downloader checks range responses, ETag, member size, selection hash, and per-file SHA-256. It writes only to the new attempt directory.

At the first progress snapshot, 21/1,306 videos (1.24 GB) were verified.
Acquisition is active; this is not a completed media corpus or a training run.
The current C: free-space reading was approximately 172 GB after acquisition
had begun; no existing artifacts were deleted.

## Implementation now in branch

Branch `codex/r2-holoassist-textfree`, commit `ecca87093e4c610f4258fbc9c6d6e2fd35cdbc60`
adds a text-free causal AV LSTM readout. It receives frozen video and audio
features plus a typed task-category ID shared by the modality arms; it does
not receive natural-language text or a text embedding. The video-only and
audio-only arms mask the other sensor, and CPU tests verify invariance to
changes in that masked input. A new internal-validation event path reads only
the official train split member. The feature packer now supports a video/audio
cache with no text bank, and the extractor accepts the new attempt's SHA-bound
selection explicitly while preserving the old pinned-selection default.

Checks for the first pushed commit: 2,443 passed / 11 skipped in the full CPU
suite; focused Ruff passed; GitHub Actions run
[39203925505](https://github.com/Unjuno/tiny-omni-decision/actions/runs/39203925505)
passed. The subsequent no-text-only-cache refinements are not yet committed or
CI-checked; their focused 42-test set currently passes.

## Remaining gates

1. Complete and hash-check all selected media.
2. Validate audio/video duration, decode the selected recordings, and build
   fresh train and development-validation action/window receipts.
3. Extract only the frozen VideoPrism and Whisper features; do not compute or
   pass text embeddings.
4. Pack, validate split identities and frozen feature hashes, then run
   video-only, audio-only, and joint arms with the pinned seed/config on the
   local machine. Any GPU work must start via HotUGPU reservation so the
   Orbitune supervisor's free-memory launch threshold remains blocked.
5. Compare NLL, Brier, ECE, Accuracy, class-balanced metrics, per-task results,
   validation prevalence, and session-group uncertainty. These are
   development results, not sealed/test results.

No new quality claim, product promotion, quantization, cloud use, or test/audit
evaluation has occurred in this attempt.

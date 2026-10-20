# Perception AV event fresh holdout protocol — 2026-10-20

Status: preregistered fixed-checkpoint inference comparison; source media
acquisition is running. This protocol adds no training, checkpoint selection,
architecture search, or sealed/test-set access.

## Question

Does the already-selected, capacity-matched joint audio+visual readout transfer
to new Perception Test TRAIN video IDs better than the already-selected
video-only readout on the same annotated event-to-object task? This is a narrow
conditional correspondence task: given an annotated audible event and visible
candidate tracks, select the track that produced the sound. It does not measure
open-world Omni decision quality.

## Frozen comparison

- Joint checkpoint: `perception-av-event-readout-joint-02/best.pt`, selected
  epoch 2, 246,658 readout parameters.
- Visual-only checkpoint: `perception-av-event-readout-video-only-matched-01/best.pt`,
  selected epoch 3, 246,115 readout parameters.
- The video-only checkpoint's instantiated scorer contains 1,225,487 total
  parameters because inactive audio/joint submodules remain in the shared
  class; only the 246,115 visual parameters are active in its fit and forward.
  The joint scorer has 246,658 total/active parameters. Checkpoint file sizes
  are 4,906,740 and 991,540 bytes respectively. These task-readout sizes are
  not the total pretrained Omni package size.
- Both checkpoints were fitted only on the original pinned features with
  manifest SHA-256
  `d92e4c91c236c69548f8fa0b58826e8afc853372ebf7a59b1e64567034f8df57`.
- The two checkpoint/result hashes are validated against their existing
  receipts before inference. Both are reloaded twice and must produce identical
  predictions on the new features.
- Candidate order is generated once by the already-frozen deterministic
  Perception AV protocol and is identical for both conditions.
- The video-only readout receives no audio feature. The joint readout receives
  the same 512-dimensional CLAP event feature and DINOv2 candidate features.
- Metrics are Accuracy, NLL, multiclass Brier, ECE and mean confidence, overall
  and by candidate count. The primary contrast is paired joint minus matched
  visual-only Accuracy/NLL with 10,000 video-cluster bootstrap draws, seed 17.
  Uniform choice is a fixed baseline.

## Data freeze and limitations

- Source revision: Perception Test
  `3938d2f1ba3a6b502025741cea4cd73c7b3bdfaf`; official TRAIN archive generation
  `1685446268139925`, pinned by byte size and ETag in the plan.
- The selection excludes all IDs in the six earlier pinned media manifests and
  all 800 video IDs in the prior 400-TRAIN/400-VALID plan. It then takes the
  first 200 eligible TRAIN video IDs under SHA-256 order
  `perception-av-event-v1:fresh-train-eval:17:{video_id}`.
- Frozen plan: `C:\CodexArtifacts\pretrained-reuse-r2\perception-av-event-fresh-holdout-plan-01\plan.json`,
  SHA-256 `887a1eea4b5a7b82d441025842465a09d0ffb33c97787073042ad1a1680f8eb3`.
  The sorted-ID digest is
  `4b549bff21445380beb989ee84905733805ce455486bf96566cc0a067a2dd895`.
- Plan generation used base commit `809c0f8226a780039a89c8066709d15d39a53f8a`
  plus the new planner file, whose SHA-256 at generation is
  `4b19256dc5f55785de85e93f9b79d7b0b40b1948334d689f505a984399b96300`.
- Plan size is 200 videos, 504 eligible events, 2,917,471,792 uncompressed
  media bytes. No replacement sampling is permitted if a media or PTS gate
  fails.
- These events/media IDs were not used in prior feature extraction or
  checkpoint fitting. However, the source annotation archives were already
  parsed for feasibility and labels were visible during deterministic
  candidate eligibility construction. This is therefore a previously unused
  TRAIN-subset transfer check, **not blind validation or a sealed audit**.
- Participant/source-recording identity is unavailable in the public
  annotations. Video-ID and exact media-byte hashes can be checked; broader
  participant independence remains UNKNOWN. A result cannot establish broad
  source-independent transfer.
- The official TEST archive remains closed. The prior official VALID results
  remain observed development validation and are not mixed into this holdout.

## Integrity gates before inference

1. Complete media acquisition from the pinned archive; check per-member
   metadata, ZIP CRC, exact byte count and SHA-256; reject repeated raw hashes
   against all previous artifacts and within this holdout.
2. Decode every selected video on CPU and verify each event's actual video PTS,
   selected frame, full audio interval coverage, and one-frame tail censoring.
3. Extract only frozen DINOv2-small and CLAP features. Verify the exact cached
   model weights/config/frontend hashes and record revision, dtype, device,
   runtime, wall time and peak CUDA allocation.
4. Use the HotUGPU cooperative guard for GPU extraction. It waits for the
   existing workload to yield and reserves VRAM before launching the child; do
   not signal or terminate any supervisor/process.
5. Evaluate the fixed best checkpoints on CPU inference, preserving every
   event prediction and probability. No result-dependent changes are allowed.

If a gate fails, keep the incomplete attempt and stop this evaluation. Do not
select easier events, change candidate counts, refit, or use these results to
modify the checkpoints.

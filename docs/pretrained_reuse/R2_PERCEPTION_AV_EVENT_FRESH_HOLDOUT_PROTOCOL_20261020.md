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

## Partial CPU PTS smoke (not a release gate)

A partial decode audit was run on the first 40 acquired plan entries while
media acquisition continued. Receipt:
`C:\CodexArtifacts\pretrained-reuse-r2\perception-av-event-fresh-holdout-pts-smoke-02\result.json`
with SHA-256
`CD8D2C813FB0B02F4F8AEDAC570099B6454C740C8F09D8C69584AD6FA5A8E062`.
It reports `PASS_PARTIAL_FRESH_HOLDOUT_PTS_SMOKE`, 40/200 videos and 97/504
events, zero events with zero/partial audio coverage, zero right-censored
events, and 1,631 distinct exact media hashes across prior manifests plus
these 40 files. The receipt pins plan SHA-256
`887a1eea4b5a7b82d441025842465a09d0ffb33c97787073042ad1a1680f8eb3`;
its media-manifest snapshot SHA-256 is
`4dfa6f3597dc6d421c799d05600eb625a6fea3fb37110faeece311a0e16d6a17`, and
the PTS audit script SHA-256 is
`3444e67c4a1acf90dc02c0cae6e72d9f0cfb2a2cc6685c61658e820db82f2568`.
This partial receipt is only a smoke test. The acquisition process was still
running when checked; GPU feature extraction and full-holdout evaluation have
not started.

## Immutable checkpoint preflight

Before evaluating any fresh-holdout features, the existing checkpoint/result
pairs were read and rehashed (2026-10-20). The joint checkpoint is 991,540
bytes, SHA-256
`be9a09f1c57c679b176d69c9a14825d4d38d053bee4b2ea5818ee108b7da8a36`, selected
epoch 2 with 246,658 trainable parameters. The matched video-only checkpoint
is 4,906,740 bytes, SHA-256
`208b283eff144f300c9d726c67eb5a063947eac87f8d4b6295764042e991cabb`, selected
epoch 3 with 246,115 active trainable parameters. Both result receipts attest
`checkpoint_reload_verified=true`, name the pinned source-feature manifest
SHA-256 `d92e4c91c236c69548f8fa0b58826e8afc853372ebf7a59b1e64567034f8df57`,
and their recorded `best.pt` hashes match the present files. No checkpoint or
fit result was modified.

A second partial CPU PTS audit covered a stable atomic manifest snapshot of 80
complete videos / 204 events while acquisition continued. Receipt:
`C:\CodexArtifacts\pretrained-reuse-r2\perception-av-event-fresh-holdout-pts-smoke-03\result.json`, SHA-256
`18513ad3d318a9b13e68cf3f1302ef6d406ab9faf4f1bb46248ffce0be81c8c`. It reports
`PASS_PARTIAL_FRESH_HOLDOUT_PTS_SMOKE`, zero incomplete audio intervals, zero
right-censored events, and 1,671 exact unique media hashes including prior
manifests. The audited media manifest snapshot SHA-256 is
`db95782c3bbaa17f2e3f7a39dbc32f91e83fd57a30f3ee50638cef9b6673dfc1`. At the
post-audit check, acquisition was still `RUNNING` and had advanced to 87 videos.
This is an incremental integrity check only; it does not authorize model feature
extraction or replace the required complete 200-video audit.

The halfway CPU audit covered a stable snapshot of 101 complete videos / 246
events. Receipt:
`C:\CodexArtifacts\pretrained-reuse-r2\perception-av-event-fresh-holdout-pts-smoke-04\result.json`, SHA-256
`5f23351dbdd59b1acfb9355e099fce401b87e76a2907d304c6a62047abfa4dd6`. It
reports `PASS_PARTIAL_FRESH_HOLDOUT_PTS_SMOKE`, zero incomplete audio intervals,
zero right-censored events, and 1,692 exact unique media hashes including
previous manifests. Snapshot manifest SHA-256:
`4a9d6cef7edbd817151fe4c13fb78051d6c320f985150ad1be1eb06488c1fee5`. The
acquirer remained live at 112/200 after this audit. As above, this partial gate
does not authorize feature extraction; the complete fixed holdout audit remains
required.

## Frozen feature-model preflight (2026-10-20)

Before feature extraction, all six locally cached DINOv2-small / CLAP model and
frontend files were rehashed against the constants enforced by the extractor;
every file matched. Exact SHA-256 values are: DINO weights
`ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1`, DINO
config `1809f83e3bdb1609a501a610ad4a742f4fd8ae44d72ca4aa0df52d1f2ac8628d`,
DINO processor `14e780d86fa1861f8751f868d7f45425b5feb55c38ca26f152ca5097ab30f828`,
CLAP audio weights
`d190141ffd119b9096d9f3e096b2285f8ce6fa401cdb666d6f53911a2517fee3`, CLAP
audio config `f7a374a0dc4907b3a30a4e9964286055929c42d932d5abd0619f7b1a20438dd3`,
and CLAP frontend `9739f58296aa6f9ac18008fd0150fb2649bc554985fbde86d0a4041c882ac753`.
The pinned runtime inventory is 22,056,576 DINO parameters and 28,190,872 CLAP
audio parameters, frozen. This read-only file check does not claim a new model
load or forward; runtime feature execution remains gated on complete media/PTS
validation and HotUGPU availability.

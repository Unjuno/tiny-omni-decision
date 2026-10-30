# VideoFDB source and task screen — 2026-10-29

## Disposition

**`REVIEW_NOT_ADMITTED` for this product-directed R2 work.** VideoFDB is a
highly relevant natural audio/video interaction evaluation candidate, but it is
manual-gated under NVIDIA's non-commercial evaluation-only agreement. It has no
training split, and its primary outcome is full-duplex response behavior scored
by an LM rubric rather than calibrated option probabilities. No license terms
were accepted and no dataset payload, annotation rows, or media were accessed.

## Pinned source and task

- Official project page: <https://research.nvidia.com/labs/amri/projects/video-fdb/>.
- Hugging Face dataset: <https://huggingface.co/datasets/nvidia/video-full-duplex-benchmark>.
- Dataset revision observed through the public Hub metadata API on 2026-10-29:
  `53a324dea6c82411d0e26d18b376b111163b906e`.
- Official card describes 237 clips from natural, locally recorded two-person
  video calls and 11 conversational dynamics, including pauses, gaze, affect,
  backchannels, and interruption. The task evaluates continuous audio/video
  interaction; its perception table includes paired AV2A and audio-only A2A
  evaluations on the same clips.
- The card lists 11 validation rows over 8 unique sample IDs and 226 test rows;
  the test data are shipped in the gated release and are not treated as a
  sealed audit here. The official project says responses are scored by an LM
  judge on conversational rubrics. This is not a training corpus or a direct
  fixed-option JSON Decision battery.

## Terms and access

The dataset card marks access as manually gated and identifies a custom NVIDIA
Evaluation License. The license grants internal use solely for evaluation and
benchmarking, explicitly prohibits commercial purposes (including model
training for commercial purposes), and prohibits identity recognition,
biometric processing, and inferring sensitive attributes. These terms do not
admit the dataset for training or product-directed model selection. No access
request was submitted and no agreement was accepted.

The public repository tree metadata exposes filenames and file sizes for the
README, license, annotation tables, and media directories. This screen did not
enumerate media files, fetch annotation files, or download media; total media
payload size remains UNKNOWN. Local C: capacity is also below the project's
2-GiB fit-output reserve.

## R2 decision

Preserve VideoFDB as a future **research-only external evaluation possibility**
if a separately approved evaluation use is later justified. It cannot train the
current candidate, cannot supply the fixed-option JSON task contract without a
separate protocol, and its published/open test contents must not be represented
as a blind audit. For current R2, do not request access, accept the license,
download the dataset, or use its scores to select the product model.

## Evidence limits

This is a metadata and terms screen, not an empirical evaluation. The
leaderboard's AV-versus-audio-only comparisons are source-reported results on
their conversational rubric; they are not evidence that the Tiny Omni Decision
candidate uses both inputs or meets any per-task quality target. Grouping of
the 237 clips by participant/conversation and exact component consent terms
were not independently audited.

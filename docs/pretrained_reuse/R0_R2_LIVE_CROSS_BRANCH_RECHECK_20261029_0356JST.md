# R0/R2 live and cross-branch recheck — 2026-10-29 03:56 JST

## Repository and PR state

- Roadmap worktree: `docs/pretrained-reuse-roadmap-20261010` at
  `b101ffd7ef020c9ef919468d2e6f82f8d5cb1328`, matching its remote branch.
- PR #15 remains open and Draft at the same head; no PR was merged, closed, or
  modified by this audit.
- GitHub reports 28 open PRs. Newer relevant source/result work is reviewed
  below; existing branches and their artifacts remain separate.

## Live resource state

Read-only Windows process inspection and `nvidia-smi` at 03:56 JST showed:

- RTX 3080 Laptop: 15,821 / 16,384 MiB allocated, 64% utilization, 63 C.
- GPU process PID 22136 (`python.exe`), parent PID 31624, started
  2026-10-28 09:12:42 local time. Its command line runs Orbitune's
  `compound_cfe_train.py` over separate MIDI train/validation indexes with
  `--steps 0`, epoch sampling, and an Orbitune-owned checkpoint/yield path.
- C: free space: 775,561,216 bytes (0.72 GiB).

The live process belongs to Orbitune. No Tiny Omni GPU workload was started;
no process signal, checkpoint, or output directory was changed. GPU memory and
the local storage reserve remain incompatible with starting a new training fit.

## Cross-branch R2 evidence

- **PR #36, DAVE/FZI-AURA (`57aa474d…`)**: DAVE is the strongest task design
  reviewed in this batch for controlled AV necessity. Its authors specify an
  AV-alignment task requiring both audio and video and include visual-only,
  audio-only, and text-only controls. Its input media include EPIC-KITCHENS
  (CC BY-NC 4.0), Ego4D (agreement-bound academic/research use), and ESC-50
  (CC BY-NC 3.0); therefore it remains unadmitted for product-directed work.
  FZI-AURA remains on access `HOLD`; its previously retrieved one-scene sample
  is quarantined. Do not repeat either screen or use that sample.
- **PR #35, ZOD (`3160975f…`)**: pinned SDK evidence confirms common UUID
  co-reference for camera 2D and optional LiDAR 3D boxes. Access-request
  ambiguity, parent-drive grouping, pretraining overlap, and measured
  both-input benefit remain unresolved. No payload or fit is admitted.
- **PR #34, GOOSE (`2a6a022f…`)**: the corrected annotation mapping retains
  valid instance ID zero and filters to instance-labeled classes. Shared 2D/3D
  IDs remain inconsistent, and there is still no non-oracle inference-time
  task target. GOOSE remains unadmitted; the corrected source report supersedes
  the previous mapping counts but does not justify another validation or fit.

These open PRs are observations, not changes to this branch or authorization to
access their data. They do not close the R2-M gate. DAVE is task-promising but
rights-blocked; seek a naturally recorded AV task with explicit usable media
rights, source-group isolation, and a fixed both-required protocol before any
new fusion fit.

## Audit method and limits

Read-only commands used Git status/revision and `ls-remote`, public GitHub REST
PR/files/commit/check APIs, `nvidia-smi`, and Windows CIM process metadata. PR
content was read from pinned public PR heads; no checkout, fetch, media access,
training, evaluation, or sealed-audit access was performed. Snapshot time is
local machine time; external GitHub timestamps are UTC. This snapshot does not
replace any preserved older inventory.

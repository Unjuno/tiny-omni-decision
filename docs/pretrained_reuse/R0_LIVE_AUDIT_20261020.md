# R0 live-state audit — 2026-10-20

This is an additive read-only snapshot for the continuing Pretrained World
Representation Reuse goal. It does not replace the 2026-10-19 audit. The
machine-readable record includes all current dataset/source manifest hashes,
open PR head/base SHAs, worktree heads and dirty counts, GPU telemetry, live
Orbitune process IDs, free disk, and the current HoloAssist sensor-run hashes:
[`R0_LIVE_AUDIT_20261020.json`](R0_LIVE_AUDIT_20261020.json).

## Repository and preserved work

- Repository: `Unjuno/tiny-omni-decision`.
- Current research checkout: `codex/r2-holoassist-results-20261020` at
  `4a813840a2c8442b9a5aef584fd4cf74424c408b` when this snapshot was captured.
- `main` was `09246d7b1821662a6b1eba47fc08c43f9dc3863c`.
- Seven pull requests (#9–#15) were open. PR #15, the roadmap proposal,
  remained draft/open at `43974fb2f6b070e0182465edca12d061b65ed8dd`.
- All 20 registered worktrees were enumerated. Seven had local modifications
  at capture time, including the primary teacher-development checkout, QAT
  review, and two text-free video branches. Their contents were not read or
  modified as part of this snapshot. The current HoloAssist worktree contained
  only this task's in-progress changes.
- All 41 files under `manifests/` were SHA-256 recorded. No manifest was
  changed by the audit.

## Compute and artifacts

The RTX 3080 Laptop reported 15,821 MiB used of 16,384 MiB, 50% utilization,
64°C and 53.25 W. `nvidia-smi` identified compute PID 18152; Windows process
inspection tied it to Orbitune's active `compound_cfe_train` process. Its
idle-training supervisors were also present. No Tiny Omni CUDA job was
started, and no process was signaled or modified. The C: drive had
46,125,277,184 bytes free. Cloud spend was $0.

The HoloAssist sensor run, feature receipt, train/validation IMU features and
selected AVS checkpoint remain in the external
`C:/CodexArtifacts/pretrained-reuse-r2/holoassist-textfree-unused-train-recovery-02/`
tree. Their hashes and `sealed_audit_loaded: false` are recorded in the JSON
snapshot and detailed result report. No model, media, cache, Teacher, QAT or
legacy checkpoint was modified by this read-only audit.

## R0–R8 status

- **R0:** Partly open. The provisional 100–200M pre-quantization parameter
  range, <=200 MB package target, local-only compute and >=95% task aspiration
  remain engineering targets. Deployment device, locked task battery,
  task-level confidence bounds, calibration/abstention and latency/energy
  limits are not yet frozen.
- **R1:** Executable VideoPrism, DINOv2 and MOMENT checkpoints have recorded
  real-load evidence. V-JEPA 2.1 rights remain `REVIEW`; this is not a product
  checkpoint pass.
- **R2:** Measured task-specific Path A probes exist, including AV, sensor-only
  and AV+sensor HoloAssist readouts. The selected AVS checkpoint is worse than
  AV on that task. IMU ablation and timing sensitivity do not establish useful
  held-out joint quality. A single coherent four-modality model and quality
  gate remain unmet.
- **R3/R4:** Remain conditional. No newly screened source is admitted for a
  quality-selected both-required audiovisual fit; do not spend GPU time to
  manufacture a positive fusion result.
- **R5:** Feature-cache correctness and multi-query reuse have evidence;
  persistent temporal-state requirements remain open.
- **R6:** Optional and not justified by a current positive quality result.
- **R7/R8:** Quantization and locked release evaluation remain gated on task
  quality, a frozen device/task contract, and runtime/package measurement.

The next safe work is CPU/read-only R2/R0 completion and preserving all current
branches. Re-enter GPU fitting only after an independently grouped, rights-
cleared, truth-bearing task passes the repository's admission gates and the
HotGPU handoff allows the reservation.

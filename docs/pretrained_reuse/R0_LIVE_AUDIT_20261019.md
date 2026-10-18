# R0 live-state audit — 2026-10-19

This is an additive read-only repository/process snapshot. It does not replace
the original `R0_INVENTORY.json` or the 2026-10-18 live audit. Machine-readable
details, all current dataset-manifest hashes, worktree heads/status counts,
open PR heads, GPU telemetry and artifact checks are in
[`R0_LIVE_AUDIT_20261019.json`](R0_LIVE_AUDIT_20261019.json).

## Repository, roadmap PR and preserved work

- Public repository: `Unjuno/tiny-omni-decision`; local audited branch
  `codex/r2-independent-joint-task`, baseline HEAD
  `f4bfa766b3c1e3ea1da57638e41fe84d32424dd2`.
- `main` remains `09246d7b1821662a6b1eba47fc08c43f9dc3863c`.
- PR #15 remains open and draft from
  `docs/pretrained-reuse-roadmap-20261010` at
  `43974fb2f6b070e0182465edca12d061b65ed8dd`, based on the current `main`.
  The latest public API snapshot showed zero inline comments and zero submitted
  reviews. No PR was merged, closed or edited during this audit.
- Seven PRs (#9–#15) are open; #13 is the only non-draft. The machine-readable
  record stores their exact head/base SHAs and all 34 remote branch refs.
- Eighteen registered worktrees were enumerated read-only. Six pre-existing
  worktrees outside the audited branch contain user or parallel-agent changes;
  those changes were not read-modified, staged, or cleaned. The R2 worktree was
  clean at the start of this follow-up and is the only worktree changed here.
- The snapshot rehashed 38 repository dataset/source manifests. Those hashes
  are retained in the JSON so later checks can compare exact files rather than
  relying on the older 2026-10-10 inventory.

## Compute and run preservation

At the snapshot, the RTX 3080 Laptop GPU reported 0% utilization, 0 MiB used
of 16,384 MiB, and no CUDA compute PIDs. Orbitune's external supervisor,
watchdog, and training-child processes remained present; no process was
signaled, stopped, restarted, or modified. No tiny-omni training process was
identified. The previous local EuroSAT CUDA evaluation completed and released
its reservation. The user-provided HotUGPU handoff remains the required path
for any later GPU work; no GPU work was started in this audit.

Ten key EuroSAT data, feature, run and test files were confirmed present and
their SHA-256 values are recorded in the JSON. Existing VideoPrism, V-JEPA,
Teacher, QAT, AV and other research artifacts remain outside this worktree and
were not changed. The sealed audit was not opened. Cloud spend remains $0.

## Current phase status

- **R0 is still partly open.** The provisional 100–200M pre-quantization
  parameter target, <=200 MB package target, local-only compute policy and
  >=95% task-specific challenge goal are recorded. The end-user device, named
  release task battery, independent-test confidence-bound rule, calibration /
  abstention policy, and latency/energy/money release limits are not frozen.
- **R1 candidate feasibility is substantially covered, not a product pass.**
  VideoPrism-B remains an executable rights-cleared video reference; the exact
  V-JEPA 2.1 checkpoint remains `REVIEW`. DINOv2-small is now in the candidate
  inventory as an Apache-2.0 image-only feature source after strict local
  loading, real-image feature extraction and the one-time fixed-label EuroSAT
  spatial evaluation. Its selected EuroSAT macro Accuracy was 94.44% on
  validation and 90.93% on the public spatial test. It provides no temporal,
  arbitrary-text-option, or multimodal quality evidence.
- **R2 remains the active research gap.** Frozen-feature task-specific paths
  exist for text, image, audio, video and sensors, but they do not yet form a
  single validated Omni Decision system. Direct no-text image/video/audio
  evidence is task-limited; the 95% target is not met across required tasks.
  One narrow Lombard audio-to-video correspondence task has controlled evidence
  that both streams are needed for the fixed opaque-candidate mapping: absent
  either stream returns uniform four-way probabilities, and content swaps under
  unchanged IDs invalidate answers. Accuracy is 94.85% on random choices and
  84.08% on near-command hard choices, with only nine development speakers.
  This is useful task-specific joint evidence, not external OOD validation or
  broad joint quality. Before another fit, extend independent task/source
  coverage and keep unimodal/shuffle/missing controls.
- R3/R4/R6 remain conditional on R2 evidence. R5 has cache correctness results
  but persistent temporal world-state requirements remain open. R7 quantization
  and R8 locked release evaluation remain gated by task quality and a frozen
  target device/task contract.

The next action is to improve the **prospective task contract and paired-input
evidence** within R2, while preserving completed Path A results. A narrow
EuroSAT image result is not a substitute for the product objective.

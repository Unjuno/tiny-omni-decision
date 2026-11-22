# R0 live audit — 2026-11-22

## Snapshot

- Repository: `Unjuno/tiny-omni-decision`.
- Roadmap branch: `docs/pretrained-reuse-roadmap-20261010`, fast-forwarded to
  remote commit `bfb13bf6e9ac4f2ce244eb51e470afa907587380` after fetching
  `origin`. It had been seven commits behind; the fast-forward did not touch
  the two pre-existing untracked PhysicalAI split manifests in this worktree.
- PR #15 remains **open, draft, mergeable**, based on `main` at
  `09246d7b1821662a6b1eba47fc08c43f9dc3863c`. Its head is the commit above
  (GitHub connector read on 2026-11-22).
- The user's primary workspace remains on the separate
  `codex/teacher-quality-clean-dev-v1` branch with uncommitted changes. It was
  not edited, stashed, reset, or staged.
- The repository has 181 registered worktrees. No worktree, legacy run,
  checkpoint, corpus, or sealed audit was modified or opened for this snapshot.

## Local compute and storage

- C: had 137.03 GiB free of 952.66 GiB at 2026-11-22 09:51 JST.
- RTX 3080 Laptop GPU: 16 GiB total, 8,259 MiB allocated, 22% utilization,
  56 C at 09:48 JST. PID `116420` was the Orbitune `compound_cfe_train.py`
  training process, using its existing checkpoint and `--yield-sentinel`
  cooperative handoff. It was left running and untouched.
- No GPU experiment was started. CPU-only focused validation of the newly
  synchronized MiniWoB readout test module completed: `5 passed in 2.22s`.

## Current measured evidence and gates

- The newly synchronized MiniWoB DINOv2 Path A result reports 0.9609
  development Accuracy on 512 examples for one synthetic screen-only
  color-to-coordinate task. Train Accuracy was 0.9961, so a sizable gap
  remains. It is not evidence for general image understanding or other
  modalities; see `R2_MINIWOB_DINO_PATH_A_RESULTS_20261112.md`.
- The PhysicalAI video+proprioception pilot reports 0.725 Accuracy versus
  0.704 for proprioception alone, but the episode-clustered 95% interval for
  the gain is `[-0.8, +5.1]` percentage points, and corrected ECE is 0.2681.
  The source remains `REVIEW_NOT_ADMITTED` because the episode split does not
  establish independent scene/layout groups. Do not tune on its exposed
  validation or treat it as an M-2 pass; see
  `R2_PHYSICALAI_STACK_PLATFORMS_FULLPILOT_RESULTS_20261102.md` and
  `R2_PHYSICALAI_ECE_CORRIGENDUM_20261102.md`.
- The roadmap still leaves R0's complete product task battery and deployment
  latency/energy limit open. R2 M-0/M-1 coverage is partial; R2 M-2 has no
  admitted task that demonstrates reliable joint-over-best-unimodal quality.
  R3+ architecture work and quantization are not justified by the present
  evidence.
- The pinned VideoPrism-B candidate has a local-load record and explicit
  upstream rights evidence. The exact V-JEPA 2.1 checkpoint remains `REVIEW`
  for its weight-specific grant. See `R1_AUDIT.md` and
  `R1_CANDIDATE_INVENTORY.json`.

## Decision

The project has working frozen-feature Decision probes and strong narrow-task
results, but **the requested multimodal Omni Decision model is not complete**.
The immediate scientific gate remains an independently grouped, rights-cleared
task battery and an admitted multi-input task whose joint result reliably beats
the best unimodal control with acceptable calibration. Continue CPU source/task
and implementation checks while the unrelated Orbitune training process holds
the GPU; do not infer readiness from parameter counts or aggregate accuracy.

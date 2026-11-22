# R1 VideoPrism-B runtime recheck — 2026-11-21

## Scope

Rechecked the existing pinned VideoPrism-B checkpoint and official runtime on
CPU. This was a fresh load/forward integrity check only. It did not load task
media, fit a Decision head, compute task metrics, touch the existing checkpoint
or source checkout, or initialize CUDA. The RTX 3080 was idle in the system
snapshot before this CPU-only run.

## Pinned inputs

- Model: `google/videoprism-base-f16r288`
- Hub revision: `be719a406d563b66f0ac969e7c94bab8e997c81a`
- Official implementation commit: `c6eba0ece3dd49a24ffc44582966997a4a59f1d8`
- Existing local checkpoint: `C:\CodexArtifacts\pretrained-reuse-r1\videoprism-base-f16r288.npz`
- Checkpoint size and SHA-256: `457,475,296` bytes;
  `61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29`.
  Last-write time was 2026-10-10 20:07:55 UTC; it was not modified.
- Rights basis is unchanged from the prior pinned audit: Apache-2.0 on the
  exact Hub model card and CC-BY-4.0 for released materials in the pinned
  upstream README. Preserve both notices. See [R1 audit](R1_AUDIT.md).

## Fresh runtime result

Used the preserved audit environment without dependency changes: Python
3.11.16, JAX 0.6.2, Flax 0.11.1, Windows x86-64 CPU. The checkpoint SHA was
verified before load. A deterministic seed-17 synthetic float32 input shaped
`[1,8,288,288,3]` returned finite float32 features `[1,2048,768]`. The loaded
tree had 114,365,184 float32 parameters over 40 leaves. The output SHA-256 is
`df185755af2fabc6c8dcf0765743edd5cbb01cf6bb6bc7d968e469a929b9796f`, identical
to the prior pinned CPU runtime replays.

Load took 0.516 s; first compile plus forward took 7.497 s; peak working set
was 3,917,218,816 bytes; process wall time was 8.395 s. The observed device was
`TFRT_CPU_0`; this does not establish CUDA or end-device latency.

## Reproducibility artifacts

New, non-overwriting output directory:
`C:\CodexArtifacts\pretrained-reuse-r1\live-recheck-20261121-attempt-01`.
The exact command and result fields are captured in `run-metadata.json`;
start time is explicitly `UNKNOWN` because it was not recorded at launch.

- Result JSON SHA-256: `4bb8e29dd41e56515bba86be85516001b7ef87b6212f5159b32250dc53e77ba4`
- Captured stdout SHA-256: `364ddb3a2336734bfd20eeea6034ac8894d0d872ccc10fb7915ae23804996930`
- Run metadata SHA-256: `fa1cc53411a7a0679b77f4cdb3294593e6c034e9bb0e57a6b9a1675631e4aa17`
- Recheck script SHA-256: `3407fcf2577e9f3b91ac0d1673bd61adbd3b6ebe7f7c252d3079cd5d3d986645`
- Original audit script SHA-256: `a1c3d210f84bd6f6df3fc90f9dcbea31529863e12839480203a96be200c0b526`

The command was the preserved environment's `python.exe` running the copied
audit script; the only script change redirected the result to the new directory.
The previous attempt files and checkpoint remain untouched. This verifies R1
runtime integrity only. It does not establish R2 quality, joint-modality gain,
or package-size feasibility.

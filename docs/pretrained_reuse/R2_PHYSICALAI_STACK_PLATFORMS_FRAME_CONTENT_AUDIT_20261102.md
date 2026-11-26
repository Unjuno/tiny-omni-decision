# PhysicalAI stack-platforms normalized-frame overlap audit — 2026-11-02

## Result

Across the fixed pilot, the audit decoded 57,751 train frames from 700 episodes
and 20,497 validation frames from 250 episodes. Each frame was converted to
RGB, bilinearly resized to 32x32, and SHA-256 hashed. All 78,248 normalized
frame hashes were unique; train/validation exact normalized-frame overlap was
zero.

This detects exact frame equality after the stated normalization only. It does
not detect near-duplicates, shared scene/layout identity, or semantic content
overlap. The pinned publisher metadata has no scene/layout/environment ID, so
the split supports only episode-held, same-source development evidence.

## Reproduction and evidence

- Source revision: `73c00c18e7ca108e91846294b20a1d7cd5de9340`.
- Split manifest v3 SHA-256:
  `3a2849afaa3cefa414c9dbfc094976e0f1cc48ad0e9636d65573fa306d1ed967`.
- Download receipt SHA-256:
  `1a2b09ad683fe5da38fec74ac937aefaa9c5a1b2ed969285478a0f724f13a498`.
- Audit script SHA-256:
  `808614b6e7d947b52b1136838ccaff828473a25714933b242942eb6a296e3c41`.
- Runtime: Python 3.11.9, PyAV 18.1.0, Pillow 10.4.0, NumPy 2.4.6; CPU only.
- Run started 2026-11-02 14:24:41 JST, processed all 950 pilot videos, and
  completed in 389.173 seconds on CPU.
- Machine-readable output:
  `C:/CodexArtifacts/pretrained-reuse-r2/physicalai-stack-platforms-frame-overlap-20261102-attempt-02/normalized-frame-overlap.json`,
  SHA-256
  `824b529870d04f0c22758a7425431b0ebbcffc945cfa17fb891070c3f9233669`.
- No labels were read, no features were extracted, no model was fit, and no
  reserved payload was accessed. The separate Orbitune GPU process remained
  untouched.

## Task protocol

The following fixed action-choice protocol was written before feature
extraction, probe fitting, or inspection of validation labels/predictions:
[`R2_PHYSICALAI_STACK_PLATFORMS_PILOT_PROTOCOL_20261102.md`](R2_PHYSICALAI_STACK_PLATFORMS_PILOT_PROTOCOL_20261102.md).
Any result must remain a narrow episode-held development diagnostic and leave
the source `REVIEW_NOT_ADMITTED` unless the remaining scene/task/source gates
are independently satisfied.

# IntPhys temporal-input coverage audit result (2026-11-27)

## Finding

The frozen eight-frame sample has exact same-scene possible/impossible input
collisions in **9 of 90 scene groups**, covering **18 of 360** opposite-label
video pairs. With each predeclared policy at **16, 25, 32, 50, or 100 frames**,
the raw ordered PNG identity audit found **zero exact cross-label collisions**
in all 90 groups.

This identifies a concrete information-coverage limitation in the current
eight-frame sample policy. It does **not** show that additional frames improve
the learned score or decision accuracy: no VideoPrism forward, feature
extraction, model fit, threshold selection, or model evaluation was performed.
The IntPhys dev data were already observed, so this is not independent quality
evidence. PNG-byte identities are an exact-source proxy and do not reproduce
the decoded/resized VideoPrism tensor; different PNG payloads could still
normalize to the same model input.

## Frozen data and procedure

- Source archive size: `3,203,229,667` bytes.
- Publisher MD5: `6a72715007f2f3b0e9546bfa8d8fc39b`.
- Archive SHA-256: `441eebe09f6583a18dbcd9d2b6c6b5667c66bac99e3875622e0768dc22e16a0a`.
- Observed: 360 videos, 90 `(block, scene)` groups, four renditions per group,
  and two labels of each class per group.
- Frame policies were committed before the archive was read:
  8, 16, 25, 32, 50, and 100 endpoint-inclusive uniform frames from frames
  1–100. The eight-frame positions reproduce the existing
  `[1, 15, 29, 43, 57, 71, 85, 100]` policy.
- For each video, ordered hashes were constructed from exact PNG bytes without
  using labels. Labels were joined afterward only to count same-scene,
  opposite-label identity matches. All 360 opposite-label pairs were retained.
- CPU-only execution rehashed the complete pinned archive and completed in
  20.89 seconds. It did not access the hidden test.

| Frames | Selected-frame identity | Scene groups with collisions | Opposite-label pairs with collisions |
|---:|---|---:|---:|
| 8 | `3175d6f39909548fbd167c195df2dbbd0db36cf5ef2eccba666ddb85f82afdd4` | 9 | 18 |
| 16 | `96179e6c0d9e0049aacbfe8dfe550654860af7f6fd00b481a9f266d4b320f7c1` | 0 | 0 |
| 25 | `2456f5fe449dee5f4b533abd2febb661ed93bf14efe5eafc760531e0e5a568fb` | 0 | 0 |
| 32 | `2b0fa6df2d5e8e257ed1ddeaf3fdadba25296dfa90c32c4333b6d5bc731bc132` | 0 | 0 |
| 50 | `e4b1b92f92475ea4e14219283032df7cd7d9c8f946c3748db925f2e4c6aba5e9` | 0 | 0 |
| 100 | `e33f092e8bf285da419e0868f42183a76d1bed976eec99be941899ad763ac846` | 0 | 0 |

The 16-frame policy is the lowest tested count that removed all **exact PNG
sequence** collisions. That makes it a justified candidate for a single-factor
follow-up, not a chosen quality winner. Any future model comparison must keep
the same records, scene-group split, encoder revision, predictor, seed, and
selection rule, and must be labeled as a repeated development evaluation.
Before claiming benefit, first verify normalized model-input identities and
then report the fixed score against persistence with scene-group uncertainty.
No hidden/sealed test was opened.

## Reproducibility

- Protocol: [`R2_INTPHYS_FRAME_COVERAGE_PROTOCOL_20261127.md`](R2_INTPHYS_FRAME_COVERAGE_PROTOCOL_20261127.md).
- Auditor: `scripts/audit_intphys_temporal_sampling_coverage.py`, SHA-256
  `642b9786bfc81f85592c9f1b5a032a802e6b3a4aab8166e1954f5d7b39c04a2c`.
- Audit output remains outside Git at
  `C:/CodexArtifacts/pretrained-reuse-r2-local/intphys-frame-coverage-20261127-attempt-01/frame-sampling-audit-v2.json`,
  SHA-256 `d3cf42606cd382465530d46567e6e7eec56d6753b058fa889218ccd347f542e6`.
- Source commit: `6d7e5657a6352a4581aaf5606f810a383491cfbc`.
- Unit tests: 12 passed; Ruff passed for the auditor and tests.

The broader data constraint remains: IntPhys training examples are
possible-only, while its matched possible/impossible decisions are in the
already observed development set. More frame coverage can remove one source of
input aliasing but cannot create supervised impossible examples or establish
generalization. BERT remains unrelated to this video limitation; its existing
evidence is restricted to separate text tasks.

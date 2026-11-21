# MiniWoB DINO Path A saved-result replay audit (2026-11-15)

## Purpose

Recheck the preserved 2026-11-12 CPU-only Path A result against its frozen corpus, cached features, and selected checkpoint. This is artifact reproducibility verification on the already exposed development validation set, not a new quality measurement or independent evaluation.

## Finding and correction

The existing finalizer initially reported a mismatch in the `corpus.root` string. The run receipt stored the `D:\CodexArtifacts\...` path while this Windows checkout resolves the same directory through a `C:\CodexArtifacts\D-CodexArtifacts\...` junction. The corpus contents were identical. The finalizer now resolves the existing and recomputed corpus roots before comparing receipts; it does not relax any manifest, data, prediction, or checkpoint hash checks. A focused test covers path-alias equality and ensures metric changes still fail comparison.

## Replay evidence

- Frozen train / development validation rows: 512 / 512.
- Train JSONL SHA-256: `be296443c53e29b77873856c710f48ef98ef21091e113a47108dbe04e60f54b9`.
- Validation JSONL SHA-256: `2be09a9f7f9a4d85f9d642f798b69d2fc39d5fcac0b0a0f9993d4fad892a350f`.
- Corpus manifest SHA-256: `dc9c7b5c3d7c7b044e319ef108da1a84ab24eedd7a10439942f8f5d05f17eecc`.
- Effective run config SHA-256: `97fc8827ec0f322800560d9853fd10304d90bb414ae2ae14cc0e942b23a737a3`.
- Selected checkpoint SHA-256: `983366413dc2afd9f21a5411dd8793525f68342c44936d3bb2ccffe123f5c3fb`.
- Existing result JSON SHA-256 before and after replay: `f3ee110baf0fe8b9f5648f2db2418f14c4de089d00db4effdd42b10123f71685`; its timestamp remained 2026-11-12 12:02:05 UTC.
- Corpus identity, row order, screenshot hashes/pixels, targets, and all saved prediction records passed the finalizer's existing checks. Selected checkpoint prediction replay was exact (`max abs probability difference = 0`).
- Candidate-order permutation: no argmax changes; maximum option-aligned probability delta `1.385808e-6`, within the predeclared `2e-6` tolerance.
- Readout result unchanged: selected epoch 99, Accuracy `0.9609375`, NLL `0.1230855`, Brier `0.0617040`, ECE-15 `0.0506217`; black-screen and screen-permutation controls remain near four-choice chance.
- Execution was CPU-only; CUDA and cloud compute were not used. The replay did not retrain, select a new checkpoint, or modify any external run artifact.

## Verification

- `python -m pytest tests/test_miniwob_dino_readout.py -q`: 6 passed.
- `ruff check scripts/finalize_miniwob_dino_path_a.py tests/test_miniwob_dino_readout.py`: passed.
- `python scripts/finalize_miniwob_dino_path_a.py`: `EXISTING_RESULT_VERIFIED`.

## Limits and next gate

This confirms the saved result is reproducible under the current filesystem layout. It does not address the original train/validation gap, synthetic task scope, single seed, or use of the development validation set for checkpoint selection. It is not evidence for general image/video understanding, cross-modal fusion, or the product accuracy target. Preserve the result as a narrow screen-only Path A diagnostic; continue seeking an independently admitted task for broader R2 evaluation.

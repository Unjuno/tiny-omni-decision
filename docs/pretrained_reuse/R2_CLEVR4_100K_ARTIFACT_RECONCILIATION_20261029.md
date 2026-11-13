# CLEVR-4 100k frozen-readout artifact reconciliation — 2026-10-29

## Scope

Read-only validation of the completed PR #24 transfer report against its local
run metadata, frozen validation manifest, and saved per-image predictions.
No inference, fitting, calibration, selection, source download, or sealed-audit
access occurred. PR #24 and its worktree remain separate and unchanged.

## Provenance and status

- PR #24: [Record CLEVR-4 100k fixed-readout transfer results](https://github.com/Unjuno/tiny-omni-decision/pull/24),
  still Open/Draft at head `204d1f2932dc8def5a8830226a7e03926b88a6c5`, base
  `docs/pretrained-reuse-roadmap-20261010`; GitHub reports `mergeable=false`.
- The report pins the source code revision to
  `sgvaze/clevr4@cddc78fb2a8359dc958987b2c750bfdd4bfd2c73`, states CC-BY-4.0,
  and records a source archive SHA-512 re-used from its earlier immutable
  preflight. This follow-up did not re-read or re-hash the 36 GB archive.
- The candidate is the frozen VideoPrism-B + MLP image readout, bundle
  `path-a-image-videoprism-clevr4-runtime-v0`, 114,512,897 recorded resident
  parameters, and 458,143,789 payload bytes. It exceeds the provisional 200 MB
  package target and is only an image component, not an Omni runtime.
- Local output directory:
  `C:/CodexArtifacts/pretrained-reuse-r2/clevr4-100k-videoprism-transfer-eval-attempt-02/`.
  The run metadata records `status=complete`, `sealed_audit=false`, no training,
  no calibration, and no checkpoint selection.

## Independent artifact checks

Parsed all 19,988 JSONL prediction rows and independently checked:

- 19,988 unique IDs, 79,952 image-taxonomy decisions, and 19,988 rows per
  taxonomy.
- Ten unique supplied options per decision, target membership, finite
  nonnegative probabilities summing to one within `1e-5`, and saved prediction
  equal to the maximum-probability option.
- Predictions SHA-256 equals both the file hash and run metadata:
  `fdd0ff5a7ff524af68ba8c92e800999956935d617a6314f6896c70b7e96826b0`.
- Validation manifest SHA-256 equals the file hash and metadata:
  `6afb7ca40309d7a107f277bc567f8c71c855ecdff2c30eb37e24cb2e45435a9a`.
- Effective-config file SHA-256 is
  `cf3f322d637983fe563abb7752f00400aa8cf0a5c408124ca02f172a67dcf0cb`.

Recomputed Accuracy, NLL, multiclass Brier, 15-bin equal-width ECE, and mean
confidence from the saved probabilities. Values agree with `run-metadata.json`
to floating-point rounding:

| Taxonomy | N | Accuracy | NLL | Brier | ECE-15 | Mean confidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Color | 19,988 | 0.99485 | 0.02994 | 0.01034 | 0.01864 | 0.97624 |
| Texture | 19,988 | 0.90279 | 0.27437 | 0.13692 | 0.01092 | 0.89877 |
| Count | 19,988 | 0.60436 | 0.89736 | 0.49178 | 0.02179 | 0.60444 |
| Shape | 19,988 | 0.99450 | 0.02000 | 0.00926 | 0.00572 | 0.98888 |
| Macro over four taxonomies | 79,952 | 0.87412 | 0.30542 | 0.16207 | 0.01427 | 0.86708 |

## R2 interpretation

This verifies a sizable fixed-readout transfer result within one synthetic
image benchmark family. It confirms strong color/shape recognition and a
material count weakness; it does not establish natural-image quality, temporal
video, text, audio, sensor, or joint-modality quality. The validation is
already observed and must not be used to tune the readout or choose another
checkpoint. Any follow-up image work must use a newly frozen train/development
protocol and report its own independent group coverage.

This result does not meet the missing R2-M task gate and does not justify a
shared Lens, model merge, or new fusion fit. No Path A or R1 candidate was
promoted. PR #24's branch, test results, source data, and model artifacts were
not changed.

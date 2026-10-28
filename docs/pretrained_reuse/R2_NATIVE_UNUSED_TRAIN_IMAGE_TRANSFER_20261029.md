# Path A one-time unused-image transfer check — 2026-10-29

## Question and limits

This single CPU-only pass checks whether the fixed Path A readout transfers to
CLEVR-4 image groups that were present in the original training partition but
were not consumed by its selected checkpoint. It is not an independent
validation set, a sealed audit, or a blind product-quality evaluation. The
result was not used for tuning or checkpoint selection. No training was run,
no sealed audit was loaded, and no model, corpus, or checkpoint was modified.

## Frozen inputs and partition

- Source code commit: `d2ffa45984289df695977c0125fa48c1d0d757f6`.
- Training source commit recorded by the fit receipt:
  `72c1eb47dde0c81e503d3a4d783aa5528660186e`.
- Fit-receipt SHA-256:
  `aaa1fe96a4be85a71ce146981c6b2e8b58e1308d91fc110e1ece6804b4d8048c`.
- Selected readout step: `1792`.
- Selected checkpoint SHA-256:
  `3eed6558731de60da82e75e38e599699f29663dd601f942505fec26a48b8dc1d`.
- Selected checkpoint had consumed `1,672` CLEVR-4 image groups. The
  complement contains `6,752` image groups and `27,008` questions. All rows
  for a held-out image group were retained together and in corpus order.
- Held-out row ID SHA-256:
  `ec9c428f41e1ab4098323b2de8850686525e170b9938a07fff51e3d197da28bb`.

The output directory is
`C:\CodexArtifacts\pretrained-reuse-r2\native-common-unused-clevr4-transfer-01`.
It contains `holdout-ids.json`, `scores.npy`, `metrics.json`, and `report.json`.
Score SHA-256: `b2347fdf07c37562f721f1160b2d27835a5f227dac998bb3b1c9471bc3742d09`.
Metrics SHA-256: `7a3841414476d34f3e1902535e7e37f809312b2d527d17c4244bbeedf0675fcc`.

## Result

| Slice | Questions | Accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| CLEVR-4 overall | 27,008 | 0.1965 | 2.1633 | 0.8645 | 0.0276 |
| Color | 6,752 | 0.1115 | 2.3226 | 0.9035 | 0.0391 |
| Count | 6,752 | 0.2490 | 1.9237 | 0.8072 | 0.0566 |
| Shape | 6,752 | 0.2984 | 2.0802 | 0.8441 | 0.1038 |
| Texture | 6,752 | 0.1272 | 2.3266 | 0.9031 | 0.0405 |

This is weak image-group-disjoint transfer for the selected readout. The
number should be interpreted in context of the task's four-choice scoring,
the fixed cached representation, and this checkpoint's limited image-group
coverage. It does not establish a general model-family ceiling or justify
changing the product gates. No follow-up tuning on these rows is authorized by
this result.

## Runtime and throughput sampling

The CPU-only evaluation completed in `49.69 s` (`543.53` rows/s overall). The
first three 256-row samples measured `377.99`, `629.32`, and `470.72` rows/s,
averaging `492.67` rows/s. That initial estimate gave about 53 seconds for the
full 27,008-row pass; actual completion was about 50 seconds. Subsequent
2,048-row windows remained roughly 508–592 rows/s. CUDA was not used.

## Integrity

The run recorded `training_performed=false`,
`checkpoint_selection_performed=false`, `sealed_audit_loaded=false`, and
`is_blind_or_independent_final_evaluation=false`. Existing v1/v2 artifacts,
corpora, and checkpoints remain unchanged.

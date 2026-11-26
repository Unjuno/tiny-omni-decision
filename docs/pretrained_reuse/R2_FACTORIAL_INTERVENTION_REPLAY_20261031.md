# Factorial intervention evaluator artifact replay — 2026-10-31

## Purpose and scope

Replayed the new saved-prediction evaluator against the preserved
`vidimu-gated-fusion-controls-03` development predictions. This checks artifact
compatibility and arithmetic only. The 39-row, three-subject validation set was
already exposed and was not used for model selection, calibration, task design,
or a new quality claim. No model inference, training, media loading, or sealed
audit access occurred.

## Provenance and result

- Source predictions: `C:/CodexArtifacts/pretrained-reuse-r2/vidimu-gated-fusion-controls-03/predictions.npz`
- Source prediction SHA-256: `c6070e4f2a2e26b4175b1869163fa4e33a02b7ea0d51cb1a6931c5c71376de51`
- Source selected checkpoint SHA-256: `d40ebf85278f5b5d3277f809c7478d4edc116832de97f1459faeacd9da365fd7`
- Source intervention-map SHA-256: `42e798f05a5d3837a5c4a48ef6915db121e865070101e761579c01cdd7bdc6b8`
- Arms replayed: matched, shuffled video, missing video, shuffled sensor, missing sensor.
- Group identity: subject prefix in each saved `Sxx_Axx_Txx` sample ID; three groups.
- New report: `C:/CodexArtifacts/pretrained-reuse-r2/factorial-intervention-replay-c6308e23-attempt-04/factorial-report.json`
- New report SHA-256: `8a5077b050090cb758545c6a2e83aba69b643e365b69bf8440ba3b334458554d`

All five arms' Accuracy, NLL, and Brier matched the preserved summary within
`rtol=1e-6, atol=5e-7`. Recomputing the original 10-bin ECE also matched its
stored value within that tolerance. The new evaluator reports 15-bin ECE by
contract, so that value is deliberately not compared to the old 10-bin ECE.
The paired intervals are recorded in the external report and remain highly
limited by only three subjects and prior checkpoint selection.

## Interpretation

The evaluator can consume a real saved intervention artifact without changing
it and reproduces the old aggregate measures. This is not an independent
multimodal result, does not prove modality necessity, and does not close R2-M.
The three-subject development predictions remain closed to further model
selection.

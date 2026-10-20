# CLEVRER temporal coverage cohort protocol (2026-10-20)

## Status

CPU-only cohort construction and integrity preflight passed. Media acquisition
is running to a separate artifact directory. No VideoPrism feature extraction
or model fitting has started in this branch yet. Orbitune still owns the local
GPU; any later CUDA job must use `scripts/run_with_hotgpu_guard.py` and wait for
its cooperative handoff.

This is an additional development cohort for the R2 text-free video path. It
does not replace the existing 512-question / 400-question reference fit or
change any Teacher artifact.

## Data selection

- Source: official CLEVRER training questions at revision
  `98b842082ba4f7c18b6b9e3f39145871782a65ef`, CC0-1.0.
- Previously unrepresented candidate video IDs come from the R2 scene-exposure
  audit and its annotation receipt. Their ID-set hash is
  `33bb4408f20ec95a64ed657da5e860588dccfd5378b95c52101ab0768baba4e8`.
- The source question file hash is
  `11181da673d223f41fb596aacfbbd3ff83d39af7f09a3210549e98cb283714b4`.
- All descriptive questions classified as temporally dependent are retained.
  Static descriptive items and all explanatory/predictive/counterfactual rows
  are outside this cohort. The latter question families remain a future task;
  this cohort only tests whether more unique temporal descriptive coverage can
  improve the established direct-video decision path.
- Scene IDs are assigned to train/validation as whole groups with seed 17 and
  an 80/20 split. Validation rows with a normalized question plus options that
  already occur in training are removed. No validation scene group is emptied.
- Natural-language question text is preserved only for source and split
  auditing. The model input remains framewise VideoPrism features, the typed
  CLEVRER program, taxonomy, and supplied options.

## Frozen CPU preflight

Preflight artifact:
`C:/CodexArtifacts/pretrained-reuse-r2/clevrer-textfree-temporal-coverage-v1-attempt-01/`

| Split | Rows | Video scenes | Question type |
|---|---:|---:|---|
| Train | 3,526 | 324 | Temporal descriptive |
| Validation | 321 | 81 | Temporal descriptive |

Normalized question/options overlap is zero. Scene overlap is zero by
construction. Validation content filtering removed 560 of the 881 initially
available validation rows, while retaining all 81 validation video groups.
The train sample-ID order SHA-256 is
`79d83adbe6a3d4eedc36a91da2a5d099e4877c272393298663f585e0cee3006e`; validation
is `b450fcfcb6004a0f1678bd0b6994e9a613a86dd38c94ef859a2db29b1655204b`.

The split scene hashes are `c6b5e43943dbc3a8b371786ccec8084c426e09f5a5be95647fad3b505bda476a`
(train) and `17332aee3f034e9918ba6c00b456b3b80fdfadec600c788eebdcc85a86624612`
(validation). Preflight binds the source questions, scene-exposure receipt and
all candidate annotation hashes. Video bytes and their hashes are not yet
included in the preflight artifact.

## Planned readout comparison

After selected media hashes are verified and the pinned VideoPrism-B framewise
features are generated, a fresh seed-17 comparison will fit the existing
`mean_mlp` and `bilstm` readouts on the same 3,526 examples and evaluate at each
epoch on the same 321 held-out development examples. The planned budget is
2,016 optimizer updates per readout (36 epochs, 56 batches per epoch at batch
size 64). Checkpoint selection uses macro taxonomy validation NLL. Scene-paired
bootstrap and video-zeroed, scene-shuffled, and temporal-reversal controls will
be reported. This validation is used for selection and is not a blind audit.

No capacity increase, sensor/audio data, sealed audit, or product promotion is
part of this experiment. Results will be added only after the run finishes and
the saved predictions and checkpoints pass integrity checks.

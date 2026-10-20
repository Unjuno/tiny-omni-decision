# CLEVRER temporal coverage cohort protocol (2026-10-20)

## Status

CPU-only cohort construction and media integrity preflight passed. Media
acquisition is complete; framewise feature extraction is running with the
existing pinned CPU JAX environment. No model fitting has started. Orbitune
still owns the local GPU; any later CUDA job must use
`scripts/run_with_hotgpu_guard.py` and wait for its cooperative handoff.

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
all candidate annotation hashes.

## Media acquisition receipt

Selected media were fetched by ZIP byte ranges from the pinned CLEVRER train
archive. The archive had 20,020 directory entries and the expected size and
ETag. The complete archive SHA-256 was not computed; each selected MP4 was
individually extracted, checked for an MP4 signature and SHA-256 hashed.

- Artifact directory:
  `C:/CodexArtifacts/pretrained-reuse-r2/clevrer-textfree-temporal-media-v1-attempt-01/`
- Verified files: 405 (324 train, 81 validation), 518,188,187 uncompressed
  bytes.
- ZIP range bytes read: 416,251,058 for train members and 104,067,467 for
  validation members; elapsed time 1,366.63 seconds.
- Train/validation media SHA-256 overlap: zero.
- Media-hash-list digests: train
  `cdd88baf0428ecdfd66db8967ab854234676036220d9403022f7fbf30cac4863`,
  validation
  `a49db7042629411b9a98505fae5495a57491aba12e215f342c6b8196a7bc421d`.
- Frozen media manifests: `train.jsonl` SHA-256
  `282174947f96cb8e33f8ada4226b4eda15a16c82ba4d45656e7cfb64b3958ed9`,
  `validation.jsonl` SHA-256
  `84f87d52bd612375314602a74295da4af7af2e8142065b47cb1612bd28efca38`.
- `fetch-report.json` SHA-256:
  `7b8deb1a4b8ca2694801233fe11c007037be11278ee2641af1a9d35bcce72425`.

The fetch report binds the prepared corpus manifests to the media-bearing
manifests. The separate CPU preflight report also binds the scene partition to
the source questions and annotation receipt.

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

# GRID Path A frozen-feature readouts (2026-10-28)

Status: **PASS for execution and artifact integrity; negative joint-video contribution on this narrow development task.** This is not an R2 exit or a general multimodal quality result.

## Protocol and frozen inputs

- Dedicated branch: `codex/r2-droid-temporal-transfer-20261028`; source commit for the frozen feature cache: `337182374682329f2685354a8fdcaa1786a55040`.
- Corrected direct GRID source split and fixed speaker-held-out cohort: 768 train utterances from 24 speakers and 256 validation utterances from eight disjoint speakers. No sealed audit was opened.
- Audio features: Whisper encoder, `[1024,384]`; video features: native temporal VideoPrism, `[1024,16,768]`; targets: six full native GRID slots `[1024,6]`.
- Frozen feature extraction ran on CPU for `2,983.772` seconds (49m 43.8s). The native video workers completed 150, 150, 150, 150, 150, 150 and 124 clips; each worker's model-state hash matched before and after. GPU was not used.
- Feature result SHA-256: `40a06e5b4a3c69b6c965a48b59d4f08a32000f7b25c0a11cd1038f37af2e87e3`.
- Feature arrays: audio `5195324aa6a81fdc5181856eec749a1dfdf06126f5481eacf859f117e4d999cf`; video `4ea332c035e2ae263b271e7f7199496caef4123c3b89c6afc23bbadc13a0eda1`; targets `ec6654fb1ed121c23f45d20c9d45c553edc927574317da8b570b65e4a86b8298`; ordered IDs `6518ad596bc0abb6e554697ebae86b2a73150e97f86e948ce59a641583380010`.

Each arm used the same 116,339-parameter readout, AdamW (`lr=0.001`, `weight_decay=0.0001`), batch 64, seed 17 and 50 epochs. Inputs were normalized from training rows only. The video branch maps each of 16 frozen temporal tokens through a 768→64 projection and GRU; audio uses a 384→64 projection; a shared 96-wide fusion layer feeds six full-vocabulary slot heads. The three arms were audio-only, video-only and joint. Each arm kept both best-validation-NLL and final-epoch checkpoints. Selection used the macro mean NLL over the six validation slots. No evaluation data was used.

The synchronized-video control re-evaluated the selected joint checkpoint after replacing each validation video with the same within-speaker position from a different held-out speaker. The feature IDs, targets and order remained fixed.

### Temporal-order sensitivity control

Without fitting another model, the selected Video-only and Joint checkpoints were replayed on the same 256 validation rows with the 16 frozen video tokens kept chronological, reversed, or independently shuffled per clip (fixed permutation seed 1702). This was a readout sensitivity probe only.

| Readout / video order | Macro Accuracy | Macro NLL | Macro Brier | Macro ECE |
|---|---:|---:|---:|---:|
| Video-only, chronological | 0.2057 | 1.8523 | 0.8108 | 0.0250 |
| Video-only, reversed | 0.1842 | 1.8504 | 0.8120 | 0.0384 |
| Video-only, shuffled | 0.1829 | 1.8508 | 0.8116 | 0.0336 |
| Joint, chronological | 0.8835 | 0.3820 | 0.1612 | 0.0365 |
| Joint, reversed | 0.8646 | 0.4533 | 0.1950 | 0.0427 |
| Joint, shuffled | 0.8763 | 0.4079 | 0.1741 | 0.0387 |

The Video-only head changed 71.5% of slot argmaxes under independent shuffling and 82.4% under reversal, while remaining near chance; therefore the GRU uses order-sensitive signals, but those signals did not support accurate slot prediction. Joint argmaxes changed only 4.0% and 7.2%, respectively, and both controls degraded NLL. This is consistent with the Joint head relying mostly on Audio, but is not proof that no video information was used. The held-out validation was already used for checkpoint selection, so these are not blind estimates.

Temporal-order result SHA-256: `3121355a5a75ed4edbbb29f7982f7c450eb65d2e73e78b0399e4b91872a2288f`; saved temporal-control predictions SHA-256: `3ed7960d54ed7bb3be96250b7305e04c9d16d80c469d8ee7a8de51e778375a88`.

## Results

Metrics below are per-slot macro averages over the six GRID decision slots; each validation slot has 256 examples. ECE uses 10 equal-width confidence bins. Values were recomputed from the reloaded best checkpoints; saved predictions had identical argmaxes and maximum absolute probability replay delta below `6.3e-7`.

| Readout | Macro Accuracy | Macro NLL | Macro Brier | Macro ECE |
|---|---:|---:|---:|---:|
| Audio-only | 0.8919 | 0.3739 | 0.1525 | 0.0318 |
| Video-only | 0.2057 | 1.8523 | 0.8108 | 0.0250 |
| Joint, matched | 0.8835 | 0.3820 | 0.1612 | 0.0365 |
| Joint, cross-speaker video mismatch | 0.8848 | 0.3904 | 0.1639 | 0.0379 |

Per-slot Accuracy / NLL / Brier / ECE:

| Slot | Audio-only | Video-only | Joint | Joint, mismatched video |
|---|---|---|---|---|
| Adverb | .9961 / .019 / .009 / .012 | .2344 / 1.382 / .748 / .038 | .9961 / .021 / .009 / .015 | .9922 / .023 / .010 / .017 |
| Color | .9453 / .138 / .076 / .026 | .2500 / 1.392 / .753 / .016 | .9336 / .158 / .089 / .028 | .9297 / .172 / .092 / .031 |
| Command | .9531 / .190 / .076 / .015 | .2969 / 1.398 / .755 / .016 | .9414 / .135 / .078 / .027 | .9453 / .149 / .087 / .028 |
| Letter | .6328 / 1.250 / .491 / .066 | .0352 / 3.242 / .962 / .014 | .6055 / 1.360 / .540 / .066 | .6133 / 1.425 / .551 / .075 |
| Number | .9258 / .234 / .105 / .020 | .0781 / 2.327 / .905 / .040 | .9297 / .217 / .090 / .029 | .9336 / .201 / .086 / .021 |
| Preposition | .8984 / .412 / .159 / .051 | .3398 / 1.372 / .742 / .026 | .8945 / .400 / .161 / .053 | .8945 / .372 / .157 / .056 |

The selected epochs were Audio 38, Video 1 and Joint 35. At the selected checkpoint, fixed-train macro Accuracy was 1.0000 for Audio and Joint versus validation 0.8919 and 0.8835; fixed-train macro NLL was 0.0104 and 0.0115 versus validation 0.3739 and 0.3820. This is a large train/validation gap. Their validation NLL rose again by epoch 50 (Audio 0.3791; Joint 0.3929) while rolling train NLL continued to fall. The video-only head did not fit even the selected train cohort well (train macro Accuracy 0.2311 / NLL 1.8305) and degraded to validation NLL 2.9322 at epoch 50. Rolling train losses are not fixed-checkpoint metrics.

Cross-speaker video replacement changed 76 of 1,536 slot argmaxes. Mean per-speaker paired matched-minus-mismatched macro Accuracy was `-0.00130`; a 10,000-replicate percentile bootstrap resampling the eight validation speaker groups gave 95% interval `[-0.00651, 0.00586]` (seed 1701). The interval includes zero and is based on validation data repeatedly used for checkpoint selection; it is only a development sensitivity diagnostic, not blind confirmatory inference.

## Interpretation and limits

This result does not support useful VideoPrism contribution to this fixed GRID closed-vocabulary speech-slot readout: video-only performance was poor, joint was slightly below audio-only, and replacing matched video did not hurt measured Accuracy. The weak joint result does not prove that temporal video models cannot help. Likely remaining explanations include a frozen general-video representation that does not preserve lip-motion information, the small 768-example fitting cohort, the readout design, or task/source limitations; this experiment does not distinguish them.

Audio performance is dominated by easy slots; the 25-way letter slot is the bottleneck. GRID is a constrained speech transcription proxy, not a general audio/video decision benchmark. Only eight independent validation speakers limit uncertainty estimates, and physical speaker identity across external corpora and product rights remain unresolved. The frozen encoders total 122,573,568 source parameters before quantization (Whisper 8,208,384 + VideoPrism 114,365,184); the readout has 116,339 trainable parameters. This is a frozen-feature research probe, not a compact end-to-end deployable package, and no quantization or product promotion was performed.

## Reproduction artifacts

The cache and readout outputs are external to Git under `C:/CodexArtifacts/pretrained-reuse-r2/` and were not modified or committed. Readout run directory:
`grid-path-a-readouts-20261028-attempt-01`.

- `result.json` SHA-256: `449e2ea2e292842c9f7bcf78e02a4dbaece6916c3f3026996a2d8511f147316c`
- `analysis.json` SHA-256: `3879dfc8ff218ed84729fe1583c5be37d1e7068b5f2e688690d3c8d415cdfda9`
- Effective readout config SHA-256: `25b25dc05420482cf1f08087212592879d46f7adbe1a0a8a423195e966092f8b`
- Fitting script SHA-256: `35d1b7468c136f124661de878bd7a5472e41a70d7c8c583496cdab3c9aaffd24`; analysis/recheck script SHA-256: `34094c9cd807eb029a4dc4a075a32da56c635e7addbe6987861da6b72d4a5a69`.
- Best checkpoints: Audio `6a3c8839173ffe62305dbc8bb70106d762fd4b9732249d4036ee93e4bd43a63c`; Video `2598f9c578188fbd0d051992aade08f9aad4ca038672396598e86e61b51dd4b1`; Joint `95cce7b4fe35df6bade0b5b779ff3d2acf41712333f5cc9a9df2aa4ba40232c1`.
- Source scripts: `scripts/fit_grid_path_a_readouts.py` and `scripts/analyze_grid_path_a_readouts.py`.
- Temporal-order replay script SHA-256: `77cb8c077e2f7f9e77106a80c9fe87292c198a1b09c8281f1d0d2b76500ea0c2` (`scripts/evaluate_grid_path_a_temporal_order.py`).

Readout fitting took 29.12 seconds on CPU; no cloud compute or GPU was used. Existing Teacher artifacts and sealed-audit data were not accessed.

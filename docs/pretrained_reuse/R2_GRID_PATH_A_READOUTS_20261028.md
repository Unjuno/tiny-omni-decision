# R2 GRID frozen-feature Path A readouts (2026-10-28)

Status: **executed and integrity-replayed; no useful video contribution and no
product-quality result.** This closes this particular development probe. Do
not repeat fitting or checkpoint selection on its eight validation speakers.

## Protocol and provenance

The frozen-feature corpus contains 768 train rows from 24 GRID speaker labels
and 256 validation rows from eight disjoint speaker labels. It uses the
corrected direct-source split. The six-slot readout protocol used seed 17,
AdamW, learning rate 0.001, weight decay 0.0001, gradient clipping 1.0, and
50 epochs; the selected checkpoint minimized mean six-slot validation NLL.
The primary control swaps validation video features across speakers. All
arms used the same feature files, cohort, and ordered IDs. Training and
evaluation ran on CPU; GPU was not used, and sealed audit was not opened.

Pinned frozen features:

- VideoPrism revision `c6eba0ece3dd49a24ffc44582966997a4a59f1d8`, 114,365,184
  parameters; checkpoint SHA-256 `61fa62eedbe8f4d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29`.
- Whisper encoder, 8,208,384 parameters; checkpoint SHA-256
  `d7f4b75d10f8ec4af79a32cc6dd20160f0c669c67bd104c1f1ab3f9d5d36a2a8`.
- Feature result SHA-256
  `40a06e5b4a3c69b6c965a48b59d4f08a32000f7b25c0a11cd1038f37af2e87e3`;
  ordered IDs SHA-256
  `6518ad596bc0abb6e554697ebae86b2a73150e97f86e948ce59a641583380010`.
- Readout result SHA-256
  `449e2ea2e292842c9f7bcf78e02a4dbaece6916c3f3026996a2d8511f147316c`;
  analysis SHA-256
  `3879dfc8ff218ed84729fe1583c5be37d1e7068b5f2e688690d3c8d415cdfda9`.
- Selected six-slot readout checkpoints: audio
  `6a3c8839173ffe62305dbc8bb70106d762fd4b9732249d4036ee93e4bd43a63c`, video
  `2598f9c578188fbd0d051992aade08f9aad4ca038672396598e86e61b51dd4b1`, joint
  `95cce7b4fe35df6bade0b5b779ff3d2acf41712333f5cc9a9df2aa4ba40232c1`.

External outputs remain in
`C:/CodexArtifacts/pretrained-reuse-r2/grid-path-a-features-20261028-attempt-01/`
and
`C:/CodexArtifacts/pretrained-reuse-r2/grid-path-a-readouts-20261028-attempt-01/`.
The feature effective config pins the extractor and worker hashes. The
readout effective config pins protocol, inputs, split, and training schedule;
its standalone runner code hash is not recorded in the saved config and is
therefore **UNKNOWN**. Do not imply stronger source-code reproducibility than
these receipts establish.

## Six-slot readout results

| Arm | Accuracy | NLL | Brier | ECE (10 bins) |
| --- | ---: | ---: | ---: | ---: |
| Audio only | 89.19% | 0.3739 | 0.1525 | 0.0318 |
| Video only | 20.57% | 1.8523 | 0.8108 | 0.0250 |
| Audio + video | 88.35% | 0.3820 | 0.1612 | 0.0365 |
| Audio + mismatched video | 88.48% | 0.3904 | 0.1639 | 0.0379 |

The joint arm did not improve over audio-only. Its accuracy was 0.13 percentage
points below the cross-speaker video-swap control; a 10,000-replicate
speaker-cluster paired bootstrap (seed 1701) gave a 95% interval of
[-0.65, +0.59] percentage points. The video-only readout was near the
six-choice chance level. This is evidence against useful video complementarity
for this readout/task, not a general limit on video models.

## Separate four-choice synchronized-video matching probe

A separate frozen-feature experiment trained new 115,457-parameter audio,
video, and joint readouts to select the synchronized clip among three
same-speaker distractors. It used 192 train quartets / 768 train queries and
64 validation quartets / 256 queries from the same 24/8 speaker partition;
seed 17, AdamW, learning rate 0.001, and up to 50 epochs with validation NLL
selection. It is not the six-slot task above.

| Condition | Accuracy | NLL | Brier | ECE (10 bins) |
| --- | ---: | ---: | ---: | ---: |
| Matched audio + video | 31.25% | 1.3702 | 0.7434 | 0.0678 |
| Both missing (uniform fallback) | 23.05% | 1.3863 | 0.7500 | 0.0195 |
| Audio missing | 25.00% | 1.3901 | 0.7519 | 0.0263 |
| Video missing | 23.05% | 1.3863 | 0.7500 | 0.0195 |
| Stale video from another quartet, same speaker | 24.22% | 1.4148 | 0.7620 | 0.0704 |
| Video offset by four frames | 32.03% | 1.3737 | 0.7452 | 0.0783 |
| Audio shuffled within quartet | 20.70% | 1.4512 | 0.7816 | 0.1108 |

Matched input exceeded the uniform fallback by 8.20 points in the
speaker-macro comparison (10,000 speaker-cluster bootstrap replicates, seed
1703; 95% interval [+2.73, +13.67] points). However, temporal offset slightly
increased accuracy, and the result is only 31.25% on four choices. It shows
some sensitivity to input correspondence but does not establish useful
temporal understanding or high-quality multimodal decisions.

The run result SHA-256 is
`22673a2967bc747182f6f12fb65a6a1940d65ddff90a2e2f958cd1217e723b95`; analysis
SHA-256 is
`ca0436c6b2d2b500d2d81284b544a21e1ecb7639d12b076ef8417a3f1810525b`.
Selected checkpoint hashes: audio
`d14499c625f53d8130707499a249760e7eb7f45577f246827c696423ea907a3c`, video
`f1ffbcf1c8edfc0e199abc7ca8066589d91c589e3c28b4e97333390a72836943`, joint
`b1d261a54a08dd52817d4d9d447bfd5ebca7c7feb7a332f77e047402d7859a8b`.
External outputs remain in
`C:/CodexArtifacts/pretrained-reuse-r2/grid-path-a-av-matching-20261028-attempt-01/`.

## Cost, limits, and disposition

Frozen extraction took 2,983.77 CPU seconds (49m44s); the two readout jobs took
29.12s and 79.14s. No GPU, cloud compute, or new download was used for these
fits. Existing feature arrays total about 49.5 MiB, with the 768D video feature
array accounting for 48.0 MiB. The matched-video experiment's effective config
records `gpu_used=false` and `sealed_audit_opened=false`.

The development validation has only eight speaker clusters and was used for
checkpoint selection, so these estimates are not blind confirmation. The
probe measures a closed-vocabulary speech-slot task and a narrow synchronized
clip-matching task, not broad AV reasoning. GRID remains research-only pending
product rights clarification. Do not tune these exposed validation sets,
promote these heads, or treat the results as meeting the roadmap's task-level
95–98% aspiration. Keep the existing Path A branches and package unchanged;
this report records external artifacts only.

**Next decision:** close this GRID Path A candidate for quality iteration.
Further R2 progress requires a different admitted, independently grouped task
whose target genuinely needs both modalities, before any new fit or large
acquisition. The earlier GRID media acquisition already completed. Its first
four transfers (1,638,400 bytes in 10.63 seconds) averaged 0.147 MiB/s pooled /
0.154 MiB/s per member; the full 97,830,912-byte range transfer took 528
seconds total (0.177 MiB/s pooled), while the complete CPU evaluation took
14.90 minutes. These are historical measurements, not a reason to repeat a
download or infer a guaranteed future wait time.

# Fixed eight-frame correspondence comparison

All200 existing development questions completed, with the same checkpoint,
candidate order and temperature1. No training, calibration or sealed data.
Each of eight existing frames uses its actual PTS-centered4.175s audio window.
Normalize each CLS vector, compute cosine for corresponding relative frame
indices, then average eight scores before softmax. This expands both audio and
visual coverage and adds averaging; it is not an isolated visual-frame effect.

| Metric | One frame | Eight frames |
| --- | ---: | ---: |
| Accuracy | .3200 | .3500 |
| NLL | 1.370076 | 1.371106 |
| Brier | .741869 | .742384 |
| ECE | .052695 | .086390 |

Paired bootstrap over50 complete candidate quartets,10000 draws,seed17,
unadjusted percentile95% intervals for eight-minus-one:

- Accuracy +.0300, interval[-.0250,.0850].
- NLL +.001030, interval[-.003079,.005224].
- Brier +.000514, interval[-.001578,.002652].

These observed development results give no clear improvement. Do not promote
or extend training based on this comparison. Correspondence is not semantic
Omni decision quality; this eight-frame adaptation is not exact upstream
sixteen-frame retrieval. Different-duration negative recordings align by relative
frame index. Existing square-resized images remain a preprocessing limitation.

Every forward was finite and exactly repeated. Saved middle-index4 features are
bitwise equal to the previous single-frame features for both modalities, which
checks that the shared condition was preserved. Elapsed feature/repeat loop
127.53s excluding load, CUDA peak allocated784,489,472bytes.

Commands: `python -m scripts.evaluate_cav_sync_eight_frames` and
`python -m scripts.compare_cav_sync_frames`.
Artifacts: `C:/CodexArtifacts/pretrained-reuse-r1/cav-mae-sync-correspondence-eight-01/`.
Features SHA-256: `4f5a995a69377a1caa41ee4e7d74812da650ec41aacf2bf54cf645e9c611f11a`.
Results retain input IDs/hashes and timing, predictions retain logits/probabilities
and targets; comparison.json contains bootstrap settings/results. No old artifact
was changed. Product weight-license status remains REVIEW.

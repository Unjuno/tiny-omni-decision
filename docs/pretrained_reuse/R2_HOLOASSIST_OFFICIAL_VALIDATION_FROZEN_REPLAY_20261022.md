# HoloAssist fixed-checkpoint replay on official validation (2026-10-22)

## Scope

This is a CPU-only replay of checkpoints already selected in the separate
`holoassist-textfree-unused-train-readout-v1` run. No fitting, calibration,
threshold change, or checkpoint selection was performed. The joint checkpoint
was fixed at step 960 by lowest NLL on that run's 264-session development split.
Audio-only and video-only controls were fixed at their already-selected steps
960 and 1,024. The purpose was to check transfer from 1,042 official-train
sessions to the 207-session official validation split.

This validation split is **not blind**: an earlier HoloAssist pilot used the
same official validation sessions to select its own readout checkpoints. These
results are an exploratory cross-run replay, not independent confirmation and
not a sealed-audit result. No result here was used to alter a model or select a
product candidate. Official test and sealed-audit payloads were not loaded.

## Identity and integrity

The candidate run used source commit
`39b1a7057c4ec0afaa04b3dbc09e1989d2fa654b`, config SHA-256
`c023a3d4a8a86ddcb3964503241691717af3bd6aa6b9fb2165110596e7ab504f`, seed 17,
VideoPrism-B plus Whisper Tiny frozen features, and a 128-unit task-conditioned
causal LSTM readout. It trained on 1,042 official-train sessions and selected
checkpoints on 264 other official-train sessions. The target cache contains
1,977 windows from 207 official-validation sessions, of which 1,896 are
labelled (544 positive, 1,352 negative). All 20 task categories occur in the
candidate vocabulary.

The candidate train and development-validation caches each have zero overlap
with the target by sample ID, session ID, and media hash. Encoder IDs,
revisions, loaded-weight hashes, and preprocessing descriptions match between
the two feature runs. The target cache SHA-256 is
`6a80d8aad25faada65039cf2586e28f2dceeed97ba5aee6bea81d5e16c15cae8`; its
receipt SHA-256 is
`050420469aad527fb4a61e7cdb46ae044cc0e694cae729b4877faf252f95d2c5`.

The candidate train and development-cache hashes are
`1c080f3dd3a661ca8e3134754dcc9ef0f88a9ac8df53f9b2fdfed5561aafd2d3` and
`2e760980460385cc615425f41e80d043fd66bc5e2d05c7ccf7d5742ee973b567`; the
candidate run-metadata SHA-256 is
`e20749b58a91b208340ebb3fd1913735da47623b57a80678df8c312f51a3e868`. The
joint, audio-only, and video-only selected checkpoint hashes are respectively
`baf41259a7ac42f4da9d1b0a2a77688191d02d3d197159237ceb6274d2dd2ed4`,
`f38d0c7b286d12e2e096f95e2f9de6ce738ea9c63a8d36cc914c96e129270658`, and
`de77cd65e3f81d424a49f9e3d0410155520798b3978e067950a458743f095e06`. The
pre-quantization resident inventory for these two encoders plus the readout is
122,906,500 parameters (114,365,184 + 8,208,384 + 332,932); this is a parameter
count, not a package-size measurement.

The source validation cache was left unchanged. A text-free evaluation copy
was written to the new external attempt directory because this candidate does
not consume text feature values. The only text-derived input is the categorical
task-type ID. Evaluation ran on CPU and took 6.27 seconds. No GPU, network
download, or new training was used.

## Results

All metrics use the same 1,896 labelled windows. ECE uses 15 bins. The task-only
baseline is a Laplace-smoothed per-task positive rate estimated from candidate
training labels; all resulting probabilities are below 0.5, so it predicts
the majority negative class.

| Fixed model | Selected step | Accuracy | Balanced accuracy | NLL | Brier | ECE | Mean confidence |
|---|---:|---:|---:|---:|---:|---:|---:|
| Joint AV | 960 | 0.7711 | 0.6873 | 0.5034 | 0.3291 | 0.0182 | 0.7768 |
| Audio only | 960 | 0.7600 | 0.6625 | 0.5061 | 0.3325 | 0.0166 | 0.7721 |
| Video only | 1,024 | 0.7194 | 0.5660 | 0.5628 | 0.3795 | 0.0228 | 0.7112 |
| Task-type prior only | n/a | 0.7131 | 0.5000 | 0.5875 | 0.3995 | 0.0311 | 0.7173 |

A paired session-cluster bootstrap used 10,000 resamples, seed 1701, sampling
the 207 sessions with replacement and retaining every labelled window from each
sampled session. Deltas below are candidate arm minus joint; intervals are
percentile 95% intervals.

| Comparison | Δ Accuracy | Δ NLL | Δ Brier |
|---|---:|---:|---:|
| Audio only − Joint AV | −0.0111 [−0.0228, +0.0005] | +0.0027 [−0.0047, +0.0101] | +0.0034 [−0.0029, +0.0097] |
| Video only − Joint AV | −0.0517 [−0.0763, −0.0277] | +0.0596 [+0.0340, +0.0847] | +0.0505 [+0.0287, +0.0715] |

On this previously exposed validation, joint AV is clearly ahead of video-only,
but its difference from audio-only is uncertain across all three paired
metrics. The pattern is consistent with audio carrying most of the decision
signal and video adding less; it does not establish that both inputs are
necessary. Accuracy remains far below the project's 95–98% target.

## Interpretation and next gate

This replay provides a cross-run check against sessions absent from the
candidate's training and checkpoint-selection split, but the older pilot's
validation selection means the dataset itself is already exposed. Do not tune
on these results or describe the intervals as fresh blind evidence. All 1,466
official-train sessions have now appeared in an earlier pilot or the larger
readout's training/development split, and all 207 official-validation sessions
were used in the earlier pilot. The official test remains sealed; HoloAssist
does not provide another untouched split for this line of experiments.

The next quality run therefore needs a separately admitted source with unused
group-disjoint development records and a predeclared task that checks whether
each claimed input is actually useful. Do not extend HoloAssist training or
change the fusion architecture based on this replay alone. The full per-task
metrics, per-example predictions, bootstrap output, evaluator source, and run
metadata are preserved externally at
`C:/CodexArtifacts/pretrained-reuse-r2/holoassist-official-validation-frozen-candidate-20261022/`.
The evaluator source SHA-256 is
`246ab7a09e176cc9130a1b9a60418a1d4c0bd6c8e3dabdeea3ffedda6663597c`; metrics
SHA-256 is
`f5f945090f971ce233aaa94430d336e7283738bd96504ae046e148accff193e5`, and the
paired-bootstrap SHA-256 is
`aafa49e8bc5dc540c0bb2a6fe7b6381f22de41c35e23d8fc638922e572734f2f`. The run
metadata hashes these files and records that no test/audit payload or fitting
was used.

# LibriSpeech train-clean-100 hard-negative readout experiment — 2026-11-17

## Question and scope

The Speech Commands Path A readout had previously shown about 26.5% accuracy
when transferred to this task family. This experiment asks whether a small
frozen-feature readout can learn the narrow decision task “which supplied
transcript exactly matches this audio?” with realistic hard distractors. It
does not test general audio QA, open-ended ASR, video, or multimodal fusion.
No sealed audit or LibriSpeech official test split was loaded. This is a
development validation result, not a blind estimate or product promotion.

## Source, split, and leakage controls

Source: OpenSLR LibriSpeech SLR12 `train-clean-100`, CC BY 4.0. The archive
was verified by official MD5 `2a93770f6d5c6c964bc36631d331a522` and SHA-256
`d4ddd1d5a6ab303066f14971d768ee43278a5f2a0aa43dc716b0e64ecbbbf6e2`.
CMUdict SHA-256 is `81917843c7f44ce2b094ac63873c2c7a4cf802040792c455ba3ca406891c3d22`;
its license file SHA-256 is `bd4ce8e44170a5f9f481310ca85c51de3c4f851a65e679b40e603b143bd3542a`.
The immutable runtime-bundle manifest SHA-256 is
`574640d88d6ddbbdb2d12f2466a248ce13c8d9eb3ac542d099b2f08981b690f4`.
All data, prepared splits, bundle, and output are under the independent C-drive
root `C:\CodexArtifacts\pretrained-reuse-r2-cdrive-20261117`; the earlier D-drive
source copy was preserved. Prepared `audio_path` values were rebased to the C
copy only; sample IDs, labels, options, order, speaker assignments, and split
membership were not changed.

| Split statistic | Train | Validation |
| --- | ---: | ---: |
| Speakers | 201 | 50 |
| Transcript rows before choice construction | 14,168 | 3,538 |
| Paired-choice examples | 13,902 | 3,467 |
| Unique audio assets used | 13,902 | 3,467 |
| Examples without three hard distractors | 41 | 14 |

Seed 17 and 20% speaker-held-out validation were fixed before training. Split
gates report zero speaker, audio-asset, and normalized-transcript overlap.
The 17,706 phonemized rows came from 28,539 transcript rows; 10,833 rows were
excluded because CMUdict could not phonemize them. The prepared train and
validation IDs and order are hash-bound in `run-report.json`.

Each example has two four-choice conditions: seeded random distractors and
hard distractors from a deterministic shortlist of at most 256 same-split
candidates, filtered by speaker/chapter, word count, character length, and
vocabulary overlap, then ranked by phoneme similarity. Both conditions use
the same target audio/transcript. This reuses training targets across the two
conditions intentionally; no target crosses the speaker split.

## Model and run

- Seed 17; four epochs; batch size 32 queries; AdamW, learning rate 0.001,
  weight decay 0.0001.
- Frozen `openai/whisper-tiny` revision `169d4a4341b33bc18d8881c4b69c2e104e1cc0af`
  audio encoder (37,760,640 parameters).
- Frozen `sentence-transformers/paraphrase-MiniLM-L3-v2` revision
  `4ca70771034acceecb2e72475f72050fcdde4ddc` text encoder (17,389,824 parameters).
- Only `FrozenFeatureCandidateScorer` was trainable (344,321 parameters).
  Total frozen+trainable component parameter count: 55,494,785 before any
  quantization. No quantization was performed.
- RTX 3080 Laptop GPU 16 GB; Python 3.11.9; PyTorch 2.6.0+cu124; CUDA 12.4;
  Transformers 5.6.2; PyAV 18.1.0; NumPy 2.4.6. Existing unrelated GPU work
  remained untouched. No cloud compute or cost.
- The process ran 4:13–4:21 UTC. GPU use was transient; most elapsed time was
  audio feature extraction. Exact process peak allocated VRAM was not captured
  (`null`); the host GPU had another process using 8,270 MiB during parts of
  the run, so a single-run peak cannot be inferred from `nvidia-smi`.

Command arguments are retained in the attempt directory's `run-report.json`.
The effective config is `effective-config.yaml`. Feature vectors, split copies,
per-example predictions, training history, checkpoints, and hashes are in the
same local output folder; the large weights/media are not committed.

## Results on the same held-speaker validation IDs

The existing Speech Commands readout and the task-adapted readout used the
same frozen audio/text encoders, exact same 3,467 target audios, same options,
and evaluation code. Four-choice chance is 25%.

| Readout | Condition | N | Accuracy | NLL | Brier | ECE (15 bins) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Existing Speech Commands | Random | 3,467 | 25.58% | 2.2252 | 1.0224 | 0.3945 |
| Existing Speech Commands | Hard | 3,467 | 25.90% | 2.1824 | 1.0139 | 0.3941 |
| LibriSpeech final epoch | Random | 3,467 | 93.94% | 0.1676 | 0.0904 | 0.0120 |
| LibriSpeech final epoch | Hard | 3,467 | 88.38% | 0.3079 | 0.1713 | 0.0235 |
| LibriSpeech final epoch | Equal-condition macro | 6,934 | 91.16% | 0.2378 | 0.1308 | 0.0178* |

`*` Macro ECE is the arithmetic mean of the two condition ECE values. ECE over
all 6,934 decisions pooled is 0.0170. The hard-negative gap is 5.57 accuracy
points, showing that random distractors overstate performance for this task.

Training CE by epoch was 0.7453, 0.2574, 0.1392, and 0.0805. Corrected
per-condition validation Accuracy/NLL by epoch was: epoch 1, random 87.42% /
0.3245 and hard 80.24% / 0.5166; epoch 2, 91.72% / 0.2144 and 85.20% /
0.3729; epoch 3, 93.60% / 0.1786 and 87.74% / 0.3172; epoch 4, 93.94% /
0.1676 and 88.38% / 0.3079.

## Evaluation bug and selection limitation

The first implementation called `_evaluate` once per condition while `_evaluate`
unconditionally calculated the macro across both conditions. The absent group
therefore yielded NaN selector NLL, so no epoch checkpoint was written; the
run failed at best-checkpoint reload after completing four epochs. The final
weights were preserved as `final-readout-pre-fix.pt`. A CPU regression test now
covers evaluating only one condition: an absent condition has count zero and
null metrics, and the active condition retains finite metrics. Four focused
tests and Ruff pass.

A corrected selector was applied post hoc to the sole saved final-epoch
checkpoint using all 6,934 fixed validation decisions (NLL 0.2377901), then
`best-readout.pt` was reloaded and produced identical metrics and predictions.
No earlier epoch weights exist, so this does **not** reconstruct a historical
best checkpoint or establish that epoch 4 was best. It is selected from the
only surviving checkpoint; future runs with the fixed code save every new best
and final weights separately. The original invalid histories were preserved
under `legacy_invalid_validation`; corrected condition metrics are retained
with the explicit fact that intermediate epoch weights were not checkpointed.

## Interpretation and next step

This is evidence that a small task-specific readout can learn LibriSpeech
transcript choice from frozen features, and that phonetic/lexical/length hard
negatives are meaningfully harder than random negatives. It is not evidence of
95–98% general audio quality, cross-domain robustness, broad supplied-option
reasoning, or product readiness. The 88.38% hard-condition result and the
single-speaker-group split are the main limitations. Do not tune repeatedly on
this validation or report it as a blind audit. A future comparison should
freeze a different task and independent source groups before testing whether
the readout generalizes beyond audiobook transcript matching.

The prior LibriSpeech precision checks remain numeric-equivalence diagnostics
only; they do not provide additional quality scores. Existing Teacher v1/v2,
ternary, sealed-audit, and product artifacts were not modified.

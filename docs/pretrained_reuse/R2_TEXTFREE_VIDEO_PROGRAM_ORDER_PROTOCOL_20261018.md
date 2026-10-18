# CLEVRER typed-program order probe protocol

## Question

The text-free CLEVRER readout mean-pools typed-program token embeddings. A prior
error audit found that temporal validation programs were usually unseen as exact
sequences, but the audit did not demonstrate that discarded order caused the
errors. This matched probe tests one narrow hypothesis: adding deterministic
sinusoidal positions before the same mean pooling improves temporal descriptive
Decision quality on previously unused scenes.

## Frozen comparison

Both arms use the same VideoPrism-B framewise features, 512 frozen training
examples from 256 scenes, seed 17, BiLSTM frame block, option scorer, optimizer,
epoch orders, and 2,048-update budget. The only difference is typed-program
pooling:

| Arm | Typed-program representation |
|---|---|
| Baseline | Mean of token embeddings; permutation-invariant |
| Candidate | Mean of token embeddings plus fixed sinusoidal position vectors |

The position vectors add no trainable parameters. Natural-language questions
remain excluded from model inputs. No xLSTM, extra tower, frame change, or
capacity increase is included.

Training uses AdamW, learning rate 0.001, weight decay 0.0001, batch size 64,
gradient clipping at 1.0, 256 epochs, and validation at each epoch. The earliest
checkpoint minimizing mean validation NLL over the present
question-type-by-taxonomy groups is selected. Best and final checkpoints are
kept separately. Both arms start from identical seed-17 initialization and
consume the same per-epoch row orders.

## Data split and leakage controls

The fixed training manifest is MIT-IBM/CLEVRER revision
`98b842082ba4f7c18b6b9e3f39145871782a65ef` (CC0-1.0), SHA-256
`5cd4e5155b0161e35b10aad2c98e14afba7d46407e635bd1c08d786e1e51765d`.
Selection data consists of 45 validation scenes with one static and one temporal
descriptive question per scene (90 total). Selection excludes every CLEVRER
scene identity found in the local prior JSONL sample/prediction inventory and
excludes normalized question/options content present in the frozen training
manifest. The raw validation-question file is pinned by SHA-256
`fdf841678a476655b906165e6ee4b0ed1516785c5ab927c8d0a4614e4e27e10d`. Only
the selected validation videos are acquired, using the pinned archive size and
ETag; the complete archive digest is not verified.

The split gate requires zero sample-ID, scene, media, and normalized
question/options overlap. Every acquired video is SHA-256 checked. The 45-scene
validation is small and task-limited; intervals will be wide. It is development
validation used for checkpoint selection, not a blind test or sealed audit.

## Selection and reporting

Report overall and per-question-type Accuracy, NLL, Brier, ECE, task taxonomy
support, selected epoch/step, train/validation metrics, update count, parameter
count, checkpoint hashes, runtime, and peak VRAM. Pair predictions by sample ID
and report scene-cluster bootstrap differences for static and temporal tasks
(5,000 resamples, seed 17, percentile 95% intervals). Video-zeroed,
scene-shuffled, and reversed-frame controls are descriptive input-use controls;
they do not prove answer correctness.

No sealed audit is loaded, no checkpoint is promoted, and no model weights,
media, or feature cache are committed. A mixed or uncertain result does not
justify expanding the architecture; the next step would require broader,
representative temporal supervision and a fresh selection split.

# HABIT train-only grouped OOF readout comparison — 2026-11-22

## Scope and protocol

The fixed HABIT train split contains 202 transition examples, 36 task groups and
66 videos. The previous six-group development validation was not used here.
Rows were connected when they shared either task-group ID or video SHA-256,
then assigned deterministically to five balanced folds (40–41 held-out rows,
7–8 task groups and 12–14 videos per fold). Assertions reject any task or video
identity crossing a fold boundary.

The four existing readouts were compared without architecture or optimizer
changes: text-only, video-only, state-only, and video+state. Each fold trained
only on its other four folds for a fixed eight epochs, using seed 17, batch 32,
AdamW at 0.001, weight decay 0.0001, and option-order randomization. Sensor
normalization was fit independently on each fold's training rows. No checkpoint
selection or early stopping was used. OOF predictions are reported at epochs
1, 2, 4 and 8. This is one train-only diagnostic run, not product evaluation.

Pinned source revision: `bae97d251907743f8fb6db7e70c61d361df4dc1e`.
Training ran on CPU only: 960 optimizer updates in 28.11 seconds; frozen text
feature extraction took 1.58 seconds. The experiment saved OOF predictions,
fold membership and per-fold learning history, but no model checkpoints.
Exact hashes and the external artifact path are in the
[run manifest](../../manifests/pretrained_reuse/habit-train-only-grouped-oof-20261122.json).

## OOF learning curve

Values below aggregate predictions from the five held-out folds over all 202
train rows and macro-average over 36 task groups. They are fixed-epoch
observations; the lowest row was not selected as a checkpoint.

| Readout | Epoch | Accuracy | NLL | Brier | ECE (15 bins) | Macro-group Accuracy | Macro-group NLL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Text | 1 | 0.4653 | 1.0247 | 0.6167 | 0.0846 | 0.5041 | 0.9961 |
| Text | 2 | 0.4752 | 1.0211 | 0.6147 | 0.0836 | 0.5199 | 0.9891 |
| Text | 4 | 0.4851 | 1.0214 | 0.6164 | 0.0767 | 0.5338 | 0.9808 |
| Text | 8 | 0.4703 | 1.0297 | 0.6222 | 0.0865 | 0.4967 | 0.9891 |
| Video | 1 | 0.4752 | 1.0262 | 0.6177 | 0.0892 | 0.5319 | 0.9975 |
| Video | 2 | 0.5050 | 1.0213 | 0.6143 | 0.0992 | 0.5708 | 0.9885 |
| Video | 4 | 0.4950 | 1.0204 | 0.6149 | 0.0839 | 0.5615 | 0.9797 |
| Video | 8 | 0.4802 | 1.0312 | 0.6220 | 0.0777 | 0.5060 | 0.9887 |
| State | 1 | 0.4752 | 1.0260 | 0.6176 | 0.0886 | 0.5319 | 0.9972 |
| State | 2 | 0.5050 | 1.0212 | 0.6143 | 0.0991 | 0.5708 | 0.9886 |
| State | 4 | 0.4950 | 1.0203 | 0.6149 | 0.0837 | 0.5615 | 0.9797 |
| State | 8 | 0.4802 | 1.0355 | 0.6243 | 0.0777 | 0.5060 | 0.9922 |
| Video + state | 1 | 0.4752 | 1.0262 | 0.6177 | 0.0888 | 0.5319 | 0.9974 |
| Video + state | 2 | 0.4901 | 1.0255 | 0.6176 | 0.0918 | 0.5546 | 0.9898 |
| Video + state | 4 | 0.4752 | 1.0189 | 0.6136 | 0.0605 | 0.5430 | 0.9793 |
| Video + state | 8 | 0.4752 | 1.0301 | 0.6208 | 0.0799 | 0.5245 | 0.9872 |

The text-only and individual-stream curves improve through epoch 4 and regress
by epoch 8 on macro-group accuracy. Paired video+state has mixed results: its
epoch-4 NLL/Brier/ECE are lowest, but sample Accuracy is below text-only and
its macro-group Accuracy gain is small. The epoch-8 paired curve is again
mixed. There is no consistent evidence that fusing both streams is better.

## Paired group uncertainty

A seed-17, 10,000-draw paired bootstrap resampled the 36 task groups at fixed
epochs. At epoch 4, candidate-minus-text macro-group Accuracy deltas were
`+0.0278` for video-only (95% percentile interval `[0.0000, 0.0833]`),
`+0.0278` for state-only (`[0.0000, 0.0833]`), and `+0.0093` for paired
(`[-0.0556, 0.0833]`). Macro-group NLL deltas were `-0.00114`, `-0.00113`,
and `-0.00153`, respectively; every 95% interval included zero. At epoch 8,
paired Accuracy delta was `+0.0278` (`[-0.0278, 0.0972]`) and NLL delta was
`-0.00185` (`[-0.00809, 0.00378]`). These wide intervals do not establish a
reliable gain. Full bootstrap outputs are preserved in the external attempt
directory and hashed in the manifest.

## Interpretation and next action

The train-only audit found many repeated instruction/candidate sets with
different target actions across videos. This grouped OOF comparison shows that
single streams can carry weak predictive signal in this task, but the current
paired readout does not demonstrate a stable fusion benefit. It does not prove
that video or state are unhelpful, that the labels are defective, or that this
model family has reached a ceiling. It is a single seed, one small source and
one fixed training schedule; OOF estimates still have sampling and training
variance, and the original development validation remains selection-affected.

Next work should prioritize diagnosing task-label/context mapping and admitting
more independent, rights-cleared task groups before widening the model or
adding gates. No HubEgo content was accessed; its use remains contingent on the
user's separate account approval. No sealed audit, quantization or product
promotion occurred.

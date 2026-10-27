# Visual LSTM residual on source-candidate train-only CV

The [fixed protocol](R2_SOURCE_VIDEO_LSTM_PROTOCOL_20261027.md) trained a
small per-candidate visual LSTM residual on the same three source-video folds
as the previous joint readout. It uses the existing frozen VideoPrism ROI frame
features and pooled EfficientAT audio, with 2,593 official-train records and
320 video groups. It does not evaluate old development, official validation,
test or sealed audit. These groups and candidate labels are not proven
independent across participants, and the task supplies oracle candidates.

The original mean/delta visual path stays intact; the additional 32-wide LSTM
receives each candidate's actual one to four chronological frame features. Its
last-minus-first hidden state is added through a zero-initialized residual
projection. A one-frame sequence produces zero temporal delta. Padding is
masked and variable sequences are packed. Tests verify original-initialization
parity and one-frame behavior. Audio remains pooled; no cross-modal clock
alignment or audio temporal model was tested.

| OOF readout | Accuracy | NLL | Brier | ECE-10 | Visible-source accuracy | None-visible accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Existing joint baseline | 0.5515 | 1.4441 | 0.5946 | 0.1240 | 0.5066 | 0.6749 |
| Visual LSTM residual | 0.5534 | 1.4505 | 0.5908 | 0.1238 | 0.5118 | 0.6676 |

LSTM minus baseline Accuracy was +0.19 percentage points, paired 95% video
bootstrap interval −1.05 to +1.46 points; NLL difference was +0.0064
(−0.0195 to +0.0326), and Brier difference −0.00387 (−0.01226 to +0.00478).
The intervals use 10,000 draws, seed 17, resampling 320 video groups. No clear
gain is established in this fixed budget. These are previously observed
train-source OOF data, the seven-epoch budget was inherited from earlier
development selection, folds overlap in training, and only one seed was used.
The LSTM also adds both capacity and temporal information; the comparison
cannot isolate their effects or rule out better temporal models/data/budgets.

The temporal head has 451,586 trainable parameters, versus 344,834 in the
baseline (+106,752, about 31%). Frozen VideoPrism and EfficientAT plus the new
head total 116,778,009 parameters, excluding the rest of the product. The
three fixed seven-epoch folds completed in 30.0 seconds on CPU; CUDA was not
initialized. All three checkpoints strictly reload; their prediction parity
has not yet been separately replayed. Original train loss fell over the seven
epochs, but only online changing-weight training loss was recorded, so it is
not a fixed-checkpoint generalization-gap measurement. No product promotion
or four-modality quality claim follows.

The run receipt at
`C:/CodexArtifacts/pretrained-reuse-r2/perception-source-video-lstm-cv-20261027-attempt-01/run.json`
SHA-256 is `298ff62ffbd7a667680c583916c9c3847f1a4d3888b732bb7c7166f79c68accf`;
the paired [comparison](C:/CodexArtifacts/pretrained-reuse-r2/perception-source-video-lstm-comparison-20261027-attempt-01/result.json)
SHA-256 is `e76dc942e1135f8e853661f9c47e0dff9c13d2b65e9bc4797ad90f3ac436facb`.
Checkpoint and prediction hashes are in the run receipt. Executed trainer
SHA-256: `e4dc331ebd9b1740d755f45d620f5fd861bfc927f49e3e2dbab52c2f23000e11`.
Targeted tests and Ruff passed; the full local suite passed 2,678 tests with
11 skipped.

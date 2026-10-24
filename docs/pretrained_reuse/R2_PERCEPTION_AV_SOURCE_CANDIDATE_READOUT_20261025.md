# Perception Test AV source-candidate readout diagnostic

## Result and scope

One seed-17, 80-epoch CPU readout run completed on fixed train/development
features. It ranks the annotated object-track candidates that produced a sound,
or selects `none_visible`, given an oracle audio interval and oracle tracks.
This is a narrow audio-to-video correspondence diagnostic, not a general
multimodal Decision model, end-to-end perception, product evaluation, or proof
that the benchmark labels establish causal sound sources.

The development partition contains 606 sound events from 80 videos. It is
group-disjoint from the 2,593 training events from 320 videos, but both sides
come from the official training source. The split is non-blind: source and
training statistics were inspected during preparation. Participant identity,
cross-video near-duplicate content, and independent test generalization are not
established. No official validation or sealed audit was read.

The selected checkpoint is epoch 7 under the predeclared lowest development
grouped-NLL rule (`min_delta=1e-4`). On the saved predictions, the joint result
is 56.93% top-1 set accuracy, NLL 1.2748, multiclass Brier 0.5039, and 10-bin
ECE 0.0675. There are 434 visible-source events and 172 `none_visible` events;
visible-source hit@1 is 50.69%, `none_visible` accuracy is 72.67%, and
visible-source candidate macro AP is 0.6655. These are exploratory development
metrics and do not meet the repository's product quality objective.

The input interventions show that both feature streams affect this readout on
this partition: zeroing audio lowers top-1 set accuracy to 36.14%, shuffling
audio across development records lowers it to 44.72%, and zeroing video lowers
it to 14.52%. Reversing the visual temporal delta lowers it to 50.17%. On the
same 570 records eligible for offset audio, source-aligned audio scores 54.56%
and NLL 1.3330, versus 48.25% and NLL 1.4793 with audio shifted by one second.
This supports input use in this readout, not a universal claim of cross-modal
reasoning or benchmark validity. The zero-input controls are out-of-distribution
interventions and are diagnostic only.

| Fixed control | N | Set Accuracy | NLL | Brier | ECE | Visible-source MAP |
|---|---:|---:|---:|---:|---:|---:|
| Joint | 606 | 56.93% | 1.2748 | 0.5039 | 0.0675 | 0.6655 |
| Zero audio | 606 | 36.14% | 1.5526 | 0.5819 | 0.0916 | 0.6236 |
| Shuffled audio | 606 | 44.72% | 1.6741 | 0.6447 | 0.1238 | 0.6344 |
| Zero video | 606 | 14.52% | 2.3638 | 0.7089 | 0.0729 | 0.2885 |
| Reversed temporal delta | 606 | 50.17% | 1.4664 | 0.5781 | 0.0854 | 0.6301 |
| Joint, offset-eligible subset | 570 | 54.56% | 1.3330 | 0.5258 | 0.0737 | 0.6663 |
| Offset audio, same subset | 570 | 48.25% | 1.4793 | 0.5740 | 0.0757 | 0.6383 |

## Learning curve and overfitting

| Epoch | Train grouped NLL | Development grouped NLL | Top-1 set Accuracy | Visible-source MAP |
|---:|---:|---:|---:|---:|
| 1 | 1.6654 | 1.4863 | 46.04% | 0.5933 |
| 3 | 1.2666 | 1.4281 | 50.33% | 0.6209 |
| 5 | 1.0860 | 1.3300 | 54.62% | 0.6556 |
| 7 selected | 0.8831 | **1.2748** | 56.93% | 0.6655 |
| 8 | 0.8027 | 1.3107 | 57.43% | 0.6694 |
| 10 | 0.6373 | 1.4641 | 54.79% | 0.6608 |
| 20 | 0.1673 | 2.7904 | 53.30% | 0.6466 |
| 40 | 0.0641 | 4.6913 | 52.31% | 0.6492 |
| 60 | 0.0669 | 6.9846 | 53.80% | 0.6486 |
| 80 final | 0.0440 | 7.6399 | 53.47% | 0.6496 |

Overfitting is clear: train loss continues falling while development NLL rises
after the selected epoch. Epoch 8 has slightly higher Accuracy and MAP but worse
NLL, so the fixed selector correctly retains epoch 7. The full 80 epochs were
run; the last checkpoint was retained separately and was not substituted for
the selected checkpoint. No early stopping or post-hoc checkpoint choice was
used.

## Readout and compute

- Trainable readout: 311,809 parameters; hidden width 128.
- Frozen features: VideoPrism base (114,365,184 parameters) and EfficientAT
  (1,961,239 parameters). The sum of only these diagnostic components is
  116,638,232 parameters. It is not a complete product parameter count and
  excludes text and standalone image decision paths.
- Features: area-weighted candidate ROI means plus chronological last-minus-
  first visual features; 2-second EfficientAT audio-window means.
- Training: seed 17, batch 32, AdamW, learning rate 1e-3, weight decay 1e-4,
  80 epochs, fixed grouped-NLL selector. Run elapsed time was 165.55 seconds
  (2026-10-24 18:42:00 to 18:44:46 UTC).
- VideoPrism feature extraction used CPU: train 2,693.19 seconds, development
  691.00 seconds. EfficientAT feature extraction used the RTX 3080 Laptop GPU
  under HotUGPU: train 122.20 seconds and development 26.49 seconds; peak
  allocated/reserved memory was 96.8/130.0 MiB for train and 96.1/127.9 MiB
  for development. Readout fitting and finalization were CPU-only; CUDA was
  uninitialized. No cloud compute or cloud cost was used.
- Environment for readout: Python 3.11.9, PyTorch 2.6.0+cu124, NumPy 2.4.6.
  Peak host RAM and CPU model were not recorded.

## Recovery and numerical checks

Attempt 01 completed fitting but failed before saving checkpoints because an
inference tensor was converted to NumPy without detaching. Attempt 02 repeated
the exact seed and configuration to preserve usable best/final states and
per-record predictions. Its training curves matched; finalization then failed
at a candidate-order check using an absolute tolerance of 1e-7. The saved
artifacts were kept. A CPU-only recovery finalizer now checks fixed input hashes,
history, selected/best state, and all saved predictions without retraining or
overwriting them.

Candidate reordering changed aligned FP32 scores by at most 1.43e-6; the check
now uses `rtol=1e-6, atol=2e-6`. Replaying saved predictions from the best state
changed probabilities by at most 4.77e-7 and scalar metrics by at most 9.54e-7.
Discrete decisions were stable on joint and other controls. The zero-video
control had 21 decision changes out of 606 under replay despite tiny score
differences, consistent with ties among near-identical inputs. The reported
zero-video metrics therefore come from the original saved prediction file; this
tie sensitivity is a warning for that ablation, not evidence of useful video
dependence by itself.

## Reproducibility artifacts

The corpus, features, model states, training history, and predictions are kept
outside Git at `C:\CodexArtifacts\pretrained-reuse-r2\`:

| Artifact | SHA-256 |
|---|---|
| Frozen feature plan | `9cf2fae738270a14e632b745b8a0f78214440eb057de76f0ba825d4413824247` |
| Train labels | `c5a8396a6b47a8635033e445c67233c76a6ecb17d27f4e6943a153fd3598600d` |
| Development labels | `701bd1ab203e54fae5fcb7cc8e5987c2d903bd23b5d2fbf736c2132e953ebcda` |
| Split manifest | `28b0bf9177d1ff2d2b89adb0c7d2916fc01180aef1d51da5b65356849ffd53aa` |
| Media acquisition manifest | `5ac15ab7ef7efda29a803f783b4a3a13fd3895ebaecbdf9c9c7e0282b9195918` |
| Train VideoPrism feature matrix | `59e7339c9c210cb942475c82032f93dd272f95d0e0a061b30a665f120b755980` |
| Development VideoPrism feature matrix | `13b5206867f67f1dddfea4a188e172653ad3d217dd0c6de65c27b3825dfc04a5` |
| Train EfficientAT feature matrix | `e92f8f62a546e147fe234e7c385857de8dbbe816ff9939060264cd65298e96fc` |
| Development EfficientAT feature matrix | `c3476ba844f311b01c81558d7b3b10e269a6c353799aadd3a3c14ebeaef3a4ec` |
| Effective config SHA-256 | `5a0869230accf504e5f67493489d936c5397d81c9a2ece2667c3ffaf89b8e96f` |
| Training history | `7255e8f4fc24047ff18a4c87aaa8925772a650f46a412b82fd4c91992523e361` |
| Selected epoch-7 state | `63390ba7532348bbd600f64ed796e92a3daf13b1f44ba0b18e83b1fd304780bf` |
| Final epoch-80 state | `e24332897044dcd42ac373490aaa1ffcbe75152bdaefde05daea30ec56ecd53b` |
| Result JSON | `C:\CodexArtifacts\pretrained-reuse-r2\perception-source-candidate-readout-attempt-02\result.json` |

Full prediction JSONL hashes and per-control metrics are in that `result.json`.
The run's original training runner SHA-256 is
`12ba61dbf855f2ccf4a21588fd3b208db7af96c8cf731b3f0cc4cdc4f8e5c587`; the
recovery finalizer records its own code hash separately. The matching source
commit before this recovery change was `d8088ad`.

## Decision and next work

This is useful evidence that the frozen AV features contain signal for the
oracle source-candidate task, but the small readout overfits quickly and remains
far below the 95–98% product aspiration. It says nothing about text or ordinary
image decisions and does not test an LSTM/xLSTM temporal core. Do not extend
training on this observed development set or treat epoch 7 as a product model.
The next quality experiment should first obtain a genuinely independent,
video-grouped evaluation source for this task or validate the target semantics,
then test whether the selected diagnostic transfers without tuning. Keep the
current small readout and frozen-feature result as the baseline. Add an LSTM or
learned fusion gate only when a preregistered, independent comparison can
separate temporal modeling from dataset and overfitting effects.

The repository-wide product objective remains at most 200M pre-quantization
parameters for text, image, audio, and video, JSON probabilities over caller
options, and high quality per defined task. This AV diagnostic neither satisfies
nor changes that objective. See the [source-candidate preflight](R2_PERCEPTION_SOURCE_CANDIDATE_PREFLIGHT_20261024.md)
for annotation semantics and earlier data-integrity gates.

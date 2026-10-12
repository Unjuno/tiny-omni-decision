# VIDIMU fixed episode comparison results

All arms used seed17, 100 full-batch AdamW epochs, lr0.001, weight decay0.01, train-only standardization and earliest minimum validation NLL. Train:130 trials/10 subjects; validation:39 trials/3 subjects. No sealed audit read.

| Arm | Best epoch | Accuracy | NLL | Brier | ECE | Head parameters |
|---|---:|---:|---:|---:|---:|---:|
| sensor | 100 | 1.000000 | 0.163704 | 0.065464 | 0.135254 | 4719 |
| video | 12 | 0.666667 | 0.776570 | 0.436735 | 0.131427 | 79911 |
| joint | 7 | 0.769231 | 0.572209 | 0.322131 | 0.119063 | 84591 |

## Learning curves

| Arm | Epoch | Train NLL | Validation NLL | Validation Accuracy |
|---|---:|---:|---:|---:|
| video | 0 | 2.849679 | 2.834426 | 0.076923 |
| video | 12 | 0.044085 | 0.776570 | 0.666667 |
| video | 25 | 0.005526 | 1.000894 | 0.666667 |
| video | 50 | 0.001882 | 1.067892 | 0.666667 |
| video | 75 | 0.001439 | 1.094735 | 0.666667 |
| video | 100 | 0.001224 | 1.105377 | 0.666667 |
| joint | 0 | 2.806624 | 2.815000 | 0.025641 |
| joint | 7 | 0.110283 | 0.572209 | 0.769231 |
| joint | 25 | 0.003190 | 0.718250 | 0.743590 |
| joint | 50 | 0.001110 | 0.757135 | 0.769231 |
| joint | 75 | 0.000863 | 0.778806 | 0.769231 |
| joint | 100 | 0.000746 | 0.784112 | 0.769231 |

Both runs completed all100 epochs; selected and final checkpoints are separate. Exact best-reload probability equality passed. Later train loss declines while validation NLL exceeds its selected minimum: overfitting evidence, not a reason to substitute final weights.

## Fixed controls

| Joint input condition | Accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|
| matched | 0.769231 | 0.572209 | 0.322131 | 0.119063 |
| shuffle_video | 0.153846 | 10.953591 | 1.356980 | 0.558714 |
| missing_video | 0.794872 | 1.656841 | 0.713710 | 0.587233 |
| shuffle_sensor | 0.564103 | 1.023075 | 0.595724 | 0.182108 |
| missing_sensor | 0.666667 | 0.818277 | 0.462117 | 0.129772 |

Shuffling either modality reduces performance, supporting dependence on both inputs. This does not establish beneficial fusion: joint NLL is worse than sensor-only. Missing-modality zeros are outside the training distribution; the higher missing-video Accuracy does not outweigh its worse NLL or establish robustness.

Subject-paired bootstrap: seed17,10000 draws, percentile95 linear quantiles, sample-weighted joint-minus-unimodal NLL.
- sensor: 0.408505; interval [0.2581235044859008, 0.6328615141243902].
- video: -0.204362; interval [-0.284569087316553, -0.0658543855372325].

Only3 validation subjects and checkpoint selection on this validation limit inference. This is not a blind audit or population-level guarantee. Fusion fails the preregistered adoption rule; no product promotion or compression follows. The sensor100% result is a narrow13-way activity task, not general Omni95-98% performance.

## Provenance and runtime

Training source: e81cad4; commands: `python -m scripts.train_vidimu_multimodal_arm --arm video` and `--arm joint`. Separate CPU float32 processes. Measured fit/evaluation/save loop: video1.413s, joint1.140s (excludes startup/loading). Cloud cost zero. VideoPrism frozen encoder114365184 parameters; resident totals: video114445095, joint114449775. Training peak RAM was not measured; feature extraction peak4.16GB is a separate measurement. No GPU training occurred.

Artifacts under `C:/CodexArtifacts/pretrained-reuse-r2/`: `vidimu-video-arm-01`, `vidimu-joint-arm-01`, retained `vidimu-sensor-arm-01`, and `vidimu-comparison-controls-01`. They include epoch0-100 predictions, targets, records, normalization, best/final weights, configs and metrics; controls include all perturbed predictions and bootstrap results. Existing legacy artifacts were not overwritten. JSON option-permutation/runtime checks for these new arms remain outstanding.

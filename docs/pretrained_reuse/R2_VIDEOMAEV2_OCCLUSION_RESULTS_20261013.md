# VideoMAEv2 fixed occlusion comparison — 2026-10-13

Execution complete; neither head is promoted. This is reused development validation, not a blind audit. Protocol: [fixed comparison](R2_VIDEOMAEV2_OCCLUSION_PROTOCOL_20261013.md).

All 356 videos / 3,082 windows extracted in 2,109.289 seconds on CPU. Admission rechecked 360 file hashes, exact train/validation routing and zero prepared-window cross-split overlap. Same 99 train / 257 validation videos, 198 / 514 questions. No legacy artifact was overwritten.

| Head/control | Accuracy | NLL | Brier | ECE | Both conditions correct |
|---|---:|---:|---:|---:|---:|
| mean/normal | 0.389105 | 1.089619 | 0.660655 | 0.020447 | 0.206226 |
| mean/zero | 0.428016 | 1.097889 | 0.666184 | 0.093294 | 0.249027 |
| mean/mismatch | 0.389105 | 1.087982 | 0.659644 | 0.020316 | 0.206226 |
| mean/reverse | 0.389105 | 1.089619 | 0.660655 | 0.020447 | 0.206226 |
| attention/normal | 0.389105 | 1.089643 | 0.660688 | 0.021040 | 0.206226 |
| attention/zero | 0.428016 | 1.097900 | 0.666191 | 0.093328 | 0.249027 |
| attention/mismatch | 0.389105 | 1.087578 | 0.659443 | 0.020943 | 0.206226 |
| attention/reverse | 0.389105 | 1.089455 | 0.660560 | 0.021559 | 0.206226 |

Both heads selected epoch 1 by fixed validation NLL; both changed-target strata have zero jointly correct initial/final pairs. Mismatched videos do not reduce Accuracy. Zero-video accuracy is not evidence of visual grounding. Larger training budgets here worsen validation NLL; this closes this fixed comparison without hyperparameter or seed search. It does not establish an encoder-family ceiling, especially with duplicated-frame inputs and pooled features.

## Learning curves and cost

### mean

Parameters: 147,712 stored / 123,008 effective. Wall time: 5.222s. Run: 9,900 presentations / 650 updates; selected checkpoint: 198 presentations. Unique train: 198 questions / 99 videos. Exact best reload passed.

| Epoch | Train Accuracy | Train NLL | Valid Accuracy | Valid NLL |
|---|---:|---:|---:|---:|
| 1 | 0.449495 | 1.049103 | 0.389105 | 1.089619 |
| 5 | 0.611111 | 0.864826 | 0.410506 | 1.151901 |
| 10 | 0.686869 | 0.678768 | 0.400778 | 1.369840 |
| 25 | 0.722222 | 0.508309 | 0.381323 | 1.854679 |
| 50 | 0.727273 | 0.434141 | 0.398833 | 2.471335 |

Candidate minus prior VideoPrism, video-cluster paired percentile95 intervals (10,000 draws, seed17):

- accuracy: [-0.017510, 0.013619]
- nll: [-0.008675, 0.009185]
- brier: [-0.007079, 0.004269]
- both_correct: [-0.011673, 0.011673]
- initial_accuracy: [-0.011673, 0.038911]
- final_accuracy: [-0.042802, 0.011673]
- ece: [-0.057066, 0.018372]
### attention

Parameters: 147,712 stored / 147,712 effective. Wall time: 7.328s. Run: 9,900 presentations / 650 updates; selected checkpoint: 198 presentations. Unique train: 198 questions / 99 videos. Exact best reload passed.

| Epoch | Train Accuracy | Train NLL | Valid Accuracy | Valid NLL |
|---|---:|---:|---:|---:|
| 1 | 0.449495 | 1.047182 | 0.389105 | 1.089643 |
| 5 | 0.656566 | 0.791639 | 0.400778 | 1.165888 |
| 10 | 0.707071 | 0.536932 | 0.404669 | 1.426877 |
| 25 | 0.742424 | 0.394307 | 0.389105 | 2.139949 |
| 50 | 0.959596 | 0.158528 | 0.371595 | 2.811064 |

Candidate minus prior VideoPrism, video-cluster paired percentile95 intervals (10,000 draws, seed17):

- accuracy: [0.000000, 0.000000]
- nll: [-0.009268, 0.009941]
- brier: [-0.006282, 0.005598]
- both_correct: [0.000000, 0.000000]
- initial_accuracy: [0.000000, 0.000000]
- final_accuracy: [0.000000, 0.000000]
- ece: [-0.016272, 0.029576]

## Reproduction and limits

Both heads use seed17, fifty epochs, the fixed selector and controls. Report runner reproduced metrics, selector and artifact hashes. No extra training, sealed audit, cloud compute or product promotion. No energy measurement; local compute only. GPU/VRAM were not used/measured for this CPU comparison.

Artifacts under `C:/CodexArtifacts/pretrained-reuse-r2/`: `perception-occlusion-videomaev2-features-02/`, `perception-occlusion-videomaev2-readout-01/`, `perception-occlusion-videomaev2-report-01.json`. Best/final weights, histories, predictions and configuration/hash ledgers retained. The two prior startup failures are preserved as described in the protocol. Source extraction commit `a12a399`; training/report commit `6b5fabe`.

The compact encoder reduces component size but has not solved conditional temporal reasoning. Per-modality and joint 95–98% product gates remain unmet; no compression or shared-Lens experiment is justified by this result alone.

## Saved-prediction collapse diagnosis

Both selected heads choose option index1 for all514 questions: target counts [157,200,157], prediction counts [0,514,0]. Thus38.91% equals200/514. Neither switches decisions between initial/final in any of257 videos, despite127 videos with different targets. Normal, mismatched and reversed-window inputs retain identical predicted labels. This is observed output collapse, not proof of an encoder capacity ceiling.

Probabilities are not constant: mismatch changes a probability by up to0.081271(mean) /0.083766(attention); reversing windows changes attention probabilities by up to0.025039. Accordingly, do not say pixels are absent from the computation. Mean initial/final probability L1 differences are0.002581/0.002605. Zero video changes40 labels but still has zero condition switches; its42.80% does not establish grounding.

Read-only diagnostic: `scripts/diagnose_occlusion_prediction_collapse.py`, pinned saved prediction hashes, output `C:/CodexArtifacts/pretrained-reuse-r2/perception-occlusion-videomaev2-collapse-01.json`. No model inference, additional fitting, checkpoint reselection or sealed evaluation. Future representation/readout work must measure conditional decision changes and matched-video evidence, not just calibrated priors.

## Train-only condition replay

Pinned best/final checkpoints were replayed on the198 training questions only, with no fitting or reselection. The297 corresponding initial/final candidate-feature pairs are never exactly equal; relative L2 distance min/median/max is0.078768/0.081116/0.082141. Numerical distinctness does not prove semantic adequacy.

| Checkpoint | Train Accuracy | Condition switches /99 | Changed-target both correct /53 |
|---|---:|---:|---:|
| Mean best | 0.449495 | 0 | 0 |
| Mean final | 0.727273 | 5 | 3 |
| Attention best | 0.449495 | 0 | 0 |
| Attention final | 0.959596 | 45 | 45 |

Attention can fit conditional answers on these training videos while its final validation NLL worsens to2.811064. The evidence rules out an absolute inability of this head to express different conditions on the training set; it does not establish useful transferable temporal features. Treat95.96% as training fit, never product or validation quality. Simply adding epochs or counting training accuracy as the goal would be misleading. The next research step must address transferable evidence/representation and data support, not a presumed condition-vector collision.

Runner: `scripts/diagnose_occlusion_train_condition.py`. Artifact: `C:/CodexArtifacts/pretrained-reuse-r2/perception-occlusion-videomaev2-train-condition-01.json`. Input admission reads existing manifests/features for integrity; prediction replay uses train only. Preserved hashes and four fixed checkpoint results; no GPU, new training, validation inference or sealed audit.

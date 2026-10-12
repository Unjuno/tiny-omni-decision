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

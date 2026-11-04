# Candidate-position balancing results

Inference completed on the fixed selected decoder; no training or selection.
Source `677d2faa0453631e11c2a28f2d8ec303c98e2f16`; pinned base/delta/input hashes in JSON.
134,515,008 resident parameters, FP32/SDPA, RTX3080Laptop16GB.
All1,024 questions/128 states in each of two paired arms; previously observed
development data, no blind/general-quality claim. Same state/question, all
cyclic option positions, temperature1, identity-aligned arithmetic mean.

| Arm | Scorer | Accuracy | NLL | Brier | ECE |
|---|---|---:|---:|---:|---:|
| base | single | 0.234375 | 1.448113 | 0.765222 | 0.034558 |
| base | rotation mean | 0.212891 | 1.443293 | 0.763040 | 0.032352 |
| option_reverse | single | 0.243164 | 1.452589 | 0.767113 | 0.025675 |
| option_reverse | rotation mean | 0.247070 | 1.442162 | 0.762561 | 0.001879 |

## Paired Accuracy difference

- base: -2.148pp; 95% interval[-5.762,+1.660]pp.
- option_reverse: +0.391pp; 95% interval[-2.734,+3.418]pp.

Paired128-state bootstrap,10,000 draws,seed17,percentile interval; no
multiple-comparison correction. Calibration and all task metrics in JSON.

Reversed-order identity consistency: 0.272461.
Cyclic averaging is not mathematically invariant to every permutation.
Rotation0 maximum probability difference from saved results: 3.42726707e-06.

## Resource cost

- Wall time 95.704s; peak allocated VRAM 660,232,192 bytes.
- Forward example counts: `{"base": {"forward_examples": 4352, "questions": 1024}, "option_reverse": {"forward_examples": 4352, "questions": 1024}}`.
- Cloud cost$0; zero optimization updates. Full probabilities and each
  individual rotation remain under `C:/CodexArtifacts/pretrained-reuse-r2/candidate-rotation-eval-01`.
- Old corpora and model files unchanged; no sealed audit access.

## Decision

Do not adopt. Base-arm Accuracy falls23.44%→21.29%, while reversed-order
Accuracy changes24.32%→24.71%. The gain is not consistent across arms.
Lower NLL alone does not establish conditional reasoning; near-uniform
probabilities can have low ECE while decisions remain near chance.
This tests one deterministic positional balancing rule, not every possible
readout or a model-family capacity limit. Do not launch another training
budget from this result alone. Preserve the stronger small BERT reference
and the four-modality95–98% objective, which remains unmet.

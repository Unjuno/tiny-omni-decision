# Selected-weight range and wording generalization

Execution PASS. Source3c73fa4. Preregistered protocol: R2_RANGE_GENERALIZATION_PROTOCOL_20261012.md. Seed97,256new states,2,048questions per arm,10,240inputs and20,480model decisions. Same selected weights throughout; no checkpoint selection or additional training. Input SHA256 `7be0e5e6398a313051527122c95b928629b774080336c4a03e628b11c188a2a6`. Previous state/content overlap0; oracle/option-position checks passed.

| Arm | Parent Acc | Candidate Acc | Difference (pp) | Paired95% interval (pp) | Parent NLL | Candidate NLL |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| control | 0.81738 | 0.82568 | 0.83 | [-0.78, 2.44] | 0.44840 | 0.40297 |
| interpolation | 0.24121 | 0.70996 | 46.88 | [44.14, 49.61] | 2.67000 | 0.68092 |
| extrapolation | 0.36182 | 0.44141 | 7.96 | [4.59, 11.28] | 1.98482 | 2.34722 |
| wording | 0.56543 | 0.42090 | -14.45 | [-16.75, -12.11] | 0.96919 | 1.55081 |
| combined | 0.31494 | 0.35254 | 3.76 | [0.73, 6.74] | 1.63262 | 2.19959 |

| Arm | Parent Brier | Candidate Brier | Parent ECE | Candidate ECE |
| --- | ---: | ---: | ---: | ---: |
| control | 0.24603 | 0.22632 | 0.03328 | 0.01305 |
| interpolation | 1.06131 | 0.37080 | 0.38829 | 0.05681 |
| extrapolation | 0.85292 | 0.83006 | 0.25554 | 0.29544 |
| wording | 0.52179 | 0.72552 | 0.03099 | 0.18890 |
| combined | 0.82862 | 0.90426 | 0.14845 | 0.28837 |

## What changed

- Expanded-range training transfers to new states under the interpolation transform:24.12% to71.00%. This is not solely training-set memorization, but remains far below95%.
- Extrapolation Accuracy improves36.18% to44.14%, while NLL and ECE worsen. Its3,072numeric values include 1224 above the training maximum1592; this is mixed rather than pure extrapolation.
- New wording is materially worse:56.54% to42.09%, with substantially worse NLL/Brier/ECE. The control remains82.57%, so this format shift matters even without changing the numbers.
- Combined changes improve Accuracy slightly but worsen NLL/Brier/ECE. More correct answers alone do not establish reliable probabilities.

Do not promote this checkpoint as a general Decision model. Evidence favors joint lexical/numeric coverage and retention checks over extending this exact numeric-only recipe. It does not identify a unique cause or justify a larger model, quantization, or a new training run without a frozen design.

## Limits and artifacts

Paired bootstrap uses256latent states,10,000resamples, seed20261012, percentile95% intervals. This is a known development comparison after observed failures; no adjustment for multiple comparisons/model selection. The candidate has extra inherited training, so it is not an equal-total-compute causal experiment. Single seed, synthetic English, four items, narrow task families. Scalar time ordering is not video reasoning. No sealed audit read, no95%multimodal claim.

Local inference 88.365s, peak allocated VRAM 194127360bytes, FP32/CUDA, no cloud charges. Maximum token length across both models/arms: 162; no truncation. Full per-task Accuracy/NLL/Brier/ECE and file hashes are in JSON. Inputs: `C:\CodexArtifacts\pretrained-reuse-r2\range-generalization-inputs-01`; outputs: `C:\CodexArtifacts\pretrained-reuse-r2\range-generalization-eval-01`. Legacy artifacts preserved.

## Reproduction

```powershell
python -m scripts.range_generalization prepare --root C:/CodexArtifacts/pretrained-reuse-r2 --output <new-input-directory>
python -m scripts.range_generalization evaluate --root C:/CodexArtifacts/pretrained-reuse-r2 --model C:/CodexArtifacts/pretrained-reuse-r1/crossencoder-minilm-l6-audit-01/model --data <input-directory> --input-sha256 7be0e5e6398a313051527122c95b928629b774080336c4a03e628b11c188a2a6 --output <new-output-directory>
```

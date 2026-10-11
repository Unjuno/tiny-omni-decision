# Fixed vocabulary-transfer comparison

Execution PASS, source13d706d. Candidate was frozen to training-selected step1664 before inference; previous lexical-range selected1408 is the comparator. No transfer-selected checkpoint, temperature fitting, extra training or discarded arms.

128 new synthetic states, 1,024 base questions; ten paired conditions / 10,240 rows per model, 8,192 distinct normalized contents. Repeated count/conditional questions do not increase independent sample size. Seed167; input SHA1a825b413e32f319dead47ccbef07ef3b36f2c391ca2c197a694aaeb4ed54dd2. This is failure-guided development, not blind evaluation.

Families:0 minimum/maximum;1 smallest/largest;2 lowest/highest;3 least/greatest;4 domain terms cheapest/priciest, safest/riskiest, earliest/latest. Families0–3 are now training vocabulary; family4 also changes field naming, so it is not a pure word substitution. Expanded range is7*x+19.

| Arm | Previous Acc | New Acc | Previous NLL | New NLL | New−previous Acc 95% CI |
| --- | ---: | ---: | ---: | ---: | --- |
| base-0 | 0.83301 | 0.83496 | 0.43557 | 0.41617 | [-0.02051, 0.02441] |
| base-1 | 0.59961 | 0.82812 | 1.05186 | 0.42145 | [0.20312, 0.25488] |
| base-2 | 0.54980 | 0.83398 | 1.24880 | 0.42005 | [0.26074, 0.30762] |
| base-3 | 0.75195 | 0.83105 | 0.62253 | 0.42273 | [0.05564, 0.10352] |
| base-4 | 0.33691 | 0.40039 | 1.45382 | 1.27573 | [0.03320, 0.09277] |
| expanded-0 | 0.73828 | 0.74219 | 0.62791 | 0.63256 | [-0.01855, 0.02637] |
| expanded-1 | 0.65820 | 0.74023 | 0.86728 | 0.63516 | [0.05469, 0.11035] |
| expanded-2 | 0.55371 | 0.73438 | 1.26320 | 0.64141 | [0.14746, 0.21484] |
| expanded-3 | 0.53223 | 0.73730 | 1.71307 | 0.63593 | [0.17383, 0.23535] |
| expanded-4 | 0.34082 | 0.34082 | 1.78617 | 1.71050 | [-0.02344, 0.02246] |

| Arm | Previous Brier | New Brier | Previous ECE | New ECE |
| --- | ---: | ---: | ---: | ---: |
| base-0 | 0.23330 | 0.23445 | 0.03521 | 0.04309 |
| base-1 | 0.51375 | 0.23846 | 0.07358 | 0.04080 |
| base-2 | 0.57688 | 0.23780 | 0.11243 | 0.03989 |
| base-3 | 0.33888 | 0.23936 | 0.06650 | 0.04562 |
| base-4 | 0.75741 | 0.69023 | 0.16176 | 0.07248 |
| expanded-0 | 0.34588 | 0.34008 | 0.04854 | 0.04162 |
| expanded-1 | 0.45683 | 0.33858 | 0.07403 | 0.05208 |
| expanded-2 | 0.58784 | 0.34391 | 0.13898 | 0.04592 |
| expanded-3 | 0.63679 | 0.34088 | 0.18928 | 0.05116 |
| expanded-4 | 0.85555 | 0.84701 | 0.26007 | 0.26242 |

## Opposing-condition correctness

| Model | Arm | Cost both/128 | Risk both/128 | Scalar time both/128 |
| --- | --- | ---: | ---: | ---: |
| parent | base-0 | 105 | 112 | 111 |
| parent | base-1 | 37 | 26 | 51 |
| parent | base-2 | 16 | 19 | 36 |
| parent | base-3 | 82 | 76 | 92 |
| parent | base-4 | 3 | 4 | 1 |
| parent | expanded-0 | 74 | 85 | 93 |
| parent | expanded-1 | 60 | 44 | 74 |
| parent | expanded-2 | 44 | 20 | 39 |
| parent | expanded-3 | 37 | 15 | 22 |
| parent | expanded-4 | 0 | 0 | 2 |
| candidate | base-0 | 104 | 108 | 105 |
| candidate | base-1 | 99 | 108 | 103 |
| candidate | base-2 | 103 | 108 | 104 |
| candidate | base-3 | 103 | 108 | 103 |
| candidate | base-4 | 6 | 13 | 10 |
| candidate | expanded-0 | 82 | 89 | 92 |
| candidate | expanded-1 | 81 | 86 | 95 |
| candidate | expanded-2 | 80 | 83 | 92 |
| candidate | expanded-3 | 82 | 84 | 92 |
| candidate | expanded-4 | 0 | 1 | 0 |

## Interpretation

Adding comparison-word coverage strongly improves previously weak trained-word families on new states/templates. It does not establish open-ended meaning transfer: domain-term Accuracy is40.04% at base range and34.08% at expanded range. Numeric expansion also reduces the trained-word arms to roughly73–74%. All conditions remain below95%. Keep the unresolved original-text, count/conditional and general multimodal requirements.

The primary baseline is the prior lexical model; both training stages share initialization and budget but selected steps differ (1408 versus1664). Per-run and selected-checkpoint exposure are reported separately in training records. No unique-layer causal attribution is established. No new probe chosen after seeing these results, no product promotion.

Accuracy intervals: paired128-state bootstrap,10,000replicates,seed20261012,percentile95%; all questions of a state kept together. Exploratory multiple arms, no simultaneous correction; repeated development/selection limits apply. NLL/Brier/ECE are point estimates. Scalar timestamps are not video reasoning.

Local FP32 evaluation 101.62s; peak CUDA allocation 194127360bytes; no cloud costs. Artifact directory `C:\CodexArtifacts\pretrained-reuse-r2\vocabulary-transfer-eval-01`. Companion JSON contains full per-task metrics, all opposing-pair counts and artifact hashes. Per-example probabilities remain local, with hashes tracked. Best model SHA `868cc435f3b70ce84d0c49781d80680ba28a1078cbcf6de25f53a58756661b13`. Existing outputs preserved.

# Coverage continuation: fixed transfer results

Execution completed; inference only. No checkpoint selection from these results.
Same128 latent states/1024 base questions,10arms/10240rows per model.
Repeated variants do not multiply independent sample size. Observed development data.
Families0–3 use trained comparison words; family4 changes only question wording.
Expanded values use7*x+19. No dropped arms, prompt search, calibration or new training.

| Arm | Old Acc | New Acc | Old NLL | New NLL | Accuracy difference95%CI |
|---|---:|---:|---:|---:|---|
| base-0 | 0.834961 | 0.843750 | 0.416171 | 0.390302 | [-0.015625,+0.033203] |
| base-1 | 0.828125 | 0.839844 | 0.421453 | 0.380910 | [-0.012695,+0.036133] |
| base-2 | 0.833984 | 0.841797 | 0.420054 | 0.384300 | [-0.016602,+0.032227] |
| base-3 | 0.831055 | 0.839844 | 0.422728 | 0.393278 | [-0.013672,+0.032227] |
| base-4 | 0.400391 | 0.401367 | 1.275728 | 1.321739 | [-0.031250,+0.032227] |
| expanded-0 | 0.742188 | 0.746094 | 0.632556 | 0.634908 | [-0.023438,+0.031250] |
| expanded-1 | 0.740234 | 0.743164 | 0.635161 | 0.643452 | [-0.023438,+0.030273] |
| expanded-2 | 0.734375 | 0.735352 | 0.641407 | 0.645616 | [-0.028320,+0.031250] |
| expanded-3 | 0.737305 | 0.735352 | 0.635931 | 0.657296 | [-0.029297,+0.026367] |
| expanded-4 | 0.340820 | 0.361328 | 1.710499 | 2.061229 | [-0.003906,+0.044922] |

| Arm | Old Brier | New Brier | Old ECE | New ECE |
|---|---:|---:|---:|---:|
| base-0 | 0.234453 | 0.220166 | 0.043088 | 0.033666 |
| base-1 | 0.238460 | 0.215608 | 0.040799 | 0.028348 |
| base-2 | 0.237803 | 0.218679 | 0.039892 | 0.024339 |
| base-3 | 0.239365 | 0.223220 | 0.045615 | 0.024014 |
| base-4 | 0.690228 | 0.708043 | 0.072482 | 0.112531 |
| expanded-0 | 0.340084 | 0.340125 | 0.041618 | 0.052818 |
| expanded-1 | 0.338576 | 0.346467 | 0.052079 | 0.058439 |
| expanded-2 | 0.343911 | 0.348195 | 0.045924 | 0.060687 |
| expanded-3 | 0.340879 | 0.353159 | 0.051164 | 0.079972 |
| expanded-4 | 0.847008 | 0.897738 | 0.262422 | 0.320425 |

## Opposite-condition pairs

| Model | Arm | Field | Both correct | Prediction changed | Pairs |
|---|---|---|---:|---:|---:|
| parent | base-0 | cost | 104 | 126 | 128 |
| parent | base-0 | risk | 108 | 128 | 128 |
| parent | base-0 | time | 105 | 128 | 128 |
| parent | base-1 | cost | 99 | 125 | 128 |
| parent | base-1 | risk | 108 | 128 | 128 |
| parent | base-1 | time | 103 | 128 | 128 |
| parent | base-2 | cost | 103 | 126 | 128 |
| parent | base-2 | risk | 108 | 128 | 128 |
| parent | base-2 | time | 104 | 128 | 128 |
| parent | base-3 | cost | 103 | 127 | 128 |
| parent | base-3 | risk | 108 | 128 | 128 |
| parent | base-3 | time | 103 | 128 | 128 |
| parent | base-4 | cost | 6 | 61 | 128 |
| parent | base-4 | risk | 13 | 67 | 128 |
| parent | base-4 | time | 10 | 64 | 128 |
| parent | expanded-0 | cost | 82 | 128 | 128 |
| parent | expanded-0 | risk | 89 | 128 | 128 |
| parent | expanded-0 | time | 92 | 128 | 128 |
| parent | expanded-1 | cost | 81 | 128 | 128 |
| parent | expanded-1 | risk | 86 | 128 | 128 |
| parent | expanded-1 | time | 95 | 128 | 128 |
| parent | expanded-2 | cost | 80 | 128 | 128 |
| parent | expanded-2 | risk | 83 | 128 | 128 |
| parent | expanded-2 | time | 92 | 128 | 128 |
| parent | expanded-3 | cost | 82 | 128 | 128 |
| parent | expanded-3 | risk | 84 | 128 | 128 |
| parent | expanded-3 | time | 92 | 128 | 128 |
| parent | expanded-4 | cost | 0 | 11 | 128 |
| parent | expanded-4 | risk | 1 | 6 | 128 |
| parent | expanded-4 | time | 0 | 8 | 128 |
| candidate | base-0 | cost | 105 | 128 | 128 |
| candidate | base-0 | risk | 110 | 128 | 128 |
| candidate | base-0 | time | 112 | 128 | 128 |
| candidate | base-1 | cost | 105 | 128 | 128 |
| candidate | base-1 | risk | 110 | 128 | 128 |
| candidate | base-1 | time | 109 | 128 | 128 |
| candidate | base-2 | cost | 104 | 128 | 128 |
| candidate | base-2 | risk | 111 | 128 | 128 |
| candidate | base-2 | time | 111 | 128 | 128 |
| candidate | base-3 | cost | 106 | 128 | 128 |
| candidate | base-3 | risk | 108 | 128 | 128 |
| candidate | base-3 | time | 109 | 128 | 128 |
| candidate | base-4 | cost | 8 | 57 | 128 |
| candidate | base-4 | risk | 15 | 70 | 128 |
| candidate | base-4 | time | 9 | 46 | 128 |
| candidate | expanded-0 | cost | 83 | 128 | 128 |
| candidate | expanded-0 | risk | 86 | 127 | 128 |
| candidate | expanded-0 | time | 89 | 125 | 128 |
| candidate | expanded-1 | cost | 80 | 128 | 128 |
| candidate | expanded-1 | risk | 82 | 127 | 128 |
| candidate | expanded-1 | time | 90 | 128 | 128 |
| candidate | expanded-2 | cost | 80 | 128 | 128 |
| candidate | expanded-2 | risk | 80 | 127 | 128 |
| candidate | expanded-2 | time | 87 | 127 | 128 |
| candidate | expanded-3 | cost | 79 | 128 | 128 |
| candidate | expanded-3 | risk | 80 | 127 | 128 |
| candidate | expanded-3 | time | 88 | 127 | 128 |
| candidate | expanded-4 | cost | 0 | 12 | 128 |
| candidate | expanded-4 | risk | 3 | 10 | 128 |
| candidate | expanded-4 | time | 0 | 4 | 128 |

## Interpretation

Training-validation improvement did not close domain-word or expanded-range fragility.
Base-family4 stays near40%; expanded-family4 reaches36.13% but NLL worsens
1.71050 to2.06123 and ECE0.26242 to0.32043. Higher Accuracy alone is insufficient.
Expanded-family0–3 probability scores generally fail to improve; preserve all regressions.
Do not adopt the checkpoint as a general reasoning fix or replace shared multimodal towers.
This is not evidence that the model family can never improve. No clean capacity ceiling.
Family4 adds domain adjectives and a field-alias preamble; state keys stay unchanged.
The next quality experiment needs a different, preregistered hypothesis; no automatic extension.

## Provenance and limits

Source `4fd19b42b747c23bcc9104e924649f7a818295fd`; checkpoint hashes and input hash in accompanying JSON.
Both full training corpora were checked for ID, normalized-state and latent-state overlap:zero.
Every symbolic target was verified; all frozen source hashes match after evaluation.
Paired128-state bootstrap10000draws/seed20261012, percentile95%, no multiplicity correction.
Repeated development evaluation and prior checkpoint selection prevent blind-test claims.
Timed evaluation94.613s; peak allocatedVRAM194127360bytes.
Cloud charges0; electricity unmeasured. Previous artifacts untouched.
Artifacts `C:/CodexArtifacts/pretrained-reuse-r2/coverage-transfer-eval-01`. No sealed audit or other modality inference.

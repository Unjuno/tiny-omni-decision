# Information-matched readout learning results

Both 1,024-update runs completed with exact best reload and frozen-weight checks.
Parent order and state evidence match; different readouts have different initial scores.
Development validation only. No sealed audit, product promotion or general95% claim.

| Arm | Step | Selector | Original Accuracy | Curriculum Accuracy |
|---|---:|---:|---:|---:|
| independent | 0 | 0.605989 | 0.635433 | 0.840820 |
| independent | 128 | 0.617668 | 0.643701 | 0.853516 |
| independent | 256 | 0.676984 | 0.631102 | 0.851562 |
| independent | 384 | 0.659568 | 0.617323 | 0.858887 |
| independent | 512 | 0.727712 | 0.612598 | 0.852539 |
| independent | 640 | 0.783277 | 0.625591 | 0.853516 |
| independent | 768 | 0.792455 | 0.623228 | 0.850098 |
| independent | 896 | 0.798465 | 0.603150 | 0.848633 |
| independent | 1024 | 0.796620 | 0.615354 | 0.834473 |

independent: best step0; selected encounters0, unique parents0. Full run8192 encounters/5120 unique parents;3072 repeated parent encounters.
Timed 601.559s; peak allocated VRAM 3191126016bytes.

| Arm | Step | Selector | Original Accuracy | Curriculum Accuracy |
|---|---:|---:|---:|---:|
| joint | 0 | 1.317061 | 0.300787 | 0.258301 |
| joint | 128 | 1.305022 | 0.453937 | 0.241699 |
| joint | 256 | 1.275148 | 0.446850 | 0.238770 |
| joint | 384 | 1.226728 | 0.531496 | 0.261719 |
| joint | 512 | 1.220146 | 0.532677 | 0.265137 |
| joint | 640 | 1.202726 | 0.564961 | 0.260254 |
| joint | 768 | 1.238872 | 0.566142 | 0.285645 |
| joint | 896 | 1.154263 | 0.575984 | 0.312500 |
| joint | 1024 | 1.102892 | 0.583071 | 0.400879 |

joint: best step1024; selected encounters8192, unique parents5120. Full run8192 encounters/5120 unique parents;3072 repeated parent encounters.
Timed 243.339s; peak allocated VRAM 964418048bytes.

## Selected and endpoint metrics

| Arm | Checkpoint | Source | Accuracy | NLL | Brier | ECE |
|---|---|---|---:|---:|---:|---:|
| independent | selected (0) | original | 0.635433 | 0.836609 | 0.471121 | 0.019318 |
| independent | selected (0) | curriculum | 0.840820 | 0.375370 | 0.214324 | 0.016823 |
| independent | endpoint (1024) | original | 0.615354 | 0.979371 | 0.527176 | 0.129652 |
| independent | endpoint (1024) | curriculum | 0.834473 | 0.613870 | 0.258960 | 0.099297 |
| joint | selected (1024) | original | 0.583071 | 0.918007 | 0.524137 | 0.039695 |
| joint | selected (1024) | curriculum | 0.400879 | 1.287777 | 0.696037 | 0.080322 |
| joint | endpoint (1024) | original | 0.583071 | 0.918007 | 0.524137 | 0.039695 |
| joint | endpoint (1024) | curriculum | 0.400879 | 1.287777 | 0.696037 | 0.080322 |

Paired state bootstrap:10,000 draws, seed17, percentile95% intervals in JSON.
Selected checkpoint metrics reuse selection validation; they are not blind estimates.
Rolling training CE averages changing examples and is not a fixed train/validation gap.
Cloud cost0; local electricity unmeasured. Full source/config/artifact hashes in JSON.
No new media evaluation. Joint readout changes layout and contextual interaction
jointly; no isolated attention causal claim. Existing artifacts remain preserved.

## Fixed reverse-order diagnostic

Natural replay exactly matches saved selected-checkpoint probabilities for both
arms and both sources (max absolute difference0). Reverse all candidates once,
then align scores to original candidate identities. No tuning or selection.

| Arm/source | Natural Accuracy | Reverse Accuracy | Changed decisions |
|---|---:|---:|---:|
| independent/original | .635433 | .635433 | 0/2540 |
| independent/curriculum | .840820 | .840820 | 0/2048 |
| joint/original | .583071 | .407874 | 1594/2540 |
| joint/curriculum | .400879 | .412598 | 1285/2048 |

Independent aligned probabilities are exactly unchanged. Joint maximum absolute
probability changes are.979874(original)/.969101(curriculum). Joint original
NLL worsens.918007->1.246457; Brier.524137->.697058; ECE.039695->.133910.
Curriculum aggregate Accuracy rises slightly despite1285 changed answers;
that is not evidence of stable identity-based decisions. All metric details and
prediction hashes are in R2_JOINT_ORDER_RESULTS_20261012.json.
Diagnostic elapsed sum107.331s, including tokenization; no training performed.

## Decision and limits

Reject this trained joint candidate. Relative to selected independent readout,
Accuracy differences are-5.24pp(original),-43.99pp(curriculum); paired-state
95% intervals exclude zero for both. Large option-order sensitivity is an
additional reason not to deploy. Do not average or tune on this diagnostic.

Joint learning is still improving at the fixed endpoint; no general capacity
ceiling or universal rejection of joint encoding is justified. CLS-trained
initialization favors the independent path; token layout, marker readout and
candidate interaction change jointly. The experiment does not isolate any one
mechanism. A future variant must explicitly address option-order robustness.

Timed training/evaluation loops601.559s(independent)/243.339s(joint): about2.47x
less elapsed time for this run, not a production latency benchmark. Peak GPU
allocation3.191GB/.964GB. Tokenization and model load are outside those timers;
bounded CPU verification work also ran concurrently. Cloud cost0.
All parameters remain22,713,601, with10,647,169trainable. Frozen hashes/reloads
passed. Same parent order and state-token accounting verified. Original state
prefixes were truncated equally (117validation questions); these are matched-
evidence scores, not the former full-context benchmark. No examples removed.

Independent selected step0 inherits coverage1792 training; zero new selected
encounters does not mean an untrained model. Existing multimodal heads were not
changed. No new media or sealed audit evaluation. Four-modality95% remains unmet.

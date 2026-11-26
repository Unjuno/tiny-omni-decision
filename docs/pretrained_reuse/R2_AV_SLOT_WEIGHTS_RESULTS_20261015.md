# Fixed TRAIN-only AV slot reliability weights

Execution and exact fresh-process refit PASS; quality adoption gates are separate.
Only six nonnegative coefficients (sum6) are fitted; encoders, original heads,
media, catalogs, native51 options and25 letters remain unchanged. No GPU,
base-model updates, cloud cost or sealed-audit access.

Source commit: `ee0a0f092544ff94963b556d7246994fe7e28a93`.

## Fixed fit and accounting

512 projected-gradient full-batch iterations; fixed final point, no development
checkpoint selection. TRAIN-only weights saved before calibration; the same
17-temperature grid chooses query temperature before development scoring.
Query temperature: 1.189207115.

Weights in ['command', 'color', 'preposition', 'letter', 'number', 'adverb'] order: `[0.9104284341783879, 0.7355415254423723, 1.0452144419138552, 0.57602596361972, 1.4390105450143034, 1.2937790898313613]`.

32640 roles/batch over4080 physical parents;32248 unique query/candidate
contexts and392 repeated contexts/batch across both policies. Source/speaker/
media/normalized-content gates and original sample order are preserved.
Iteration-level presentations: 16711680;
actual objective/gradient evaluations: 513; objective label uses:
16744320. These are repeated uses, not new assets.
398 calibration /889 development parents;36/4/9 speaker groups.

AV slice122790905 registered parameters/coefficients includes both full heads
and six added scalars (five free simplex degrees). This is not a complete
text/image/audio/video resident product or its package-size result.

## TRAIN-only raw learning curve

| Iteration | Random Acc | Random NLL | Hard Acc | Hard NLL |
|---|---:|---:|---:|---:|
| 0 | 0.99007353 | 0.04831281 | 0.94264706 | 0.20016763 |
| 1 | 0.99007353 | 0.04791274 | 0.94234069 | 0.19905789 |
| 32 | 0.99007353 | 0.04421354 | 0.94129902 | 0.19048987 |
| 64 | 0.98958333 | 0.04345711 | 0.93829657 | 0.18991965 |
| 128 | 0.98958333 | 0.04292921 | 0.93645833 | 0.18995595 |
| 256 | 0.98927696 | 0.04266184 | 0.93572304 | 0.19009495 |
| 384 | 0.98927696 | 0.04259268 | 0.93560049 | 0.19014735 |
| 512 | 0.98915441 | 0.04256825 | 0.93517157 | 0.19016916 |

## random candidates

| Variant / split | Accuracy | NLL | Brier | ECE | Mean confidence |
|---|---:|---:|---:|---:|---:|
| Parent / development | 0.94853768 | 0.14502748 | 0.07734490 | 0.02319345 | 0.92677455 |
| Candidate raw / train | 0.98915441 | 0.04256825 | 0.01999399 | 0.02035212 | 0.96955044 |
| Candidate raw / calibration | 0.92022613 | 0.22198702 | 0.12061299 | 0.01211753 | 0.91580059 |
| Candidate raw / validation | 0.94825647 | 0.13781777 | 0.07589380 | 0.00912941 | 0.94072569 |
| Candidate calibrated / train | 0.98915441 | 0.05180144 | 0.02274488 | 0.02889091 | 0.96045145 |
| Candidate calibrated / calibration | 0.92022613 | 0.22524120 | 0.12222649 | 0.02262946 | 0.89801604 |
| Candidate calibrated / validation | 0.94825647 | 0.14422053 | 0.07766726 | 0.02303016 | 0.92640570 |

Paired candidate-minus-parent intervals: whole development speaker groups,
10,000 resamples, seed17, linear percentile95%. Nine groups; quartet roles
are correlated. This observed development comparison is not a blind test.

| Metric | Difference | Lower | Upper |
|---|---:|---:|---:|
| accuracy | -0.00028121 | -0.00444939 | 0.00465116 |
| nll | -0.00080695 | -0.00907101 | 0.00839786 |
| brier | 0.00032237 | -0.00496130 | 0.00592536 |
| mean_confidence | -0.00036884 | -0.00193949 | 0.00142565 |
| ece | -0.00016329 | -0.00680610 | 0.00488410 |

| Stratum | Roles | Parent Acc | Candidate Acc | Candidate NLL |
|---|---:|---:|---:|---:|
| condition=l | 1800 | 0.95000000 | 0.95000000 | 0.12740110 |
| condition=p | 1756 | 0.94703872 | 0.94646925 | 0.16146141 |
| closest_negative_distance=1 | 4 | 0.50000000 | 0.50000000 | 1.08381221 |
| closest_negative_distance=2 | 104 | 0.75000000 | 0.74038462 | 0.50617352 |
| closest_negative_distance=3 | 556 | 0.89928058 | 0.89928058 | 0.25954229 |
| closest_negative_distance=4 | 1695 | 0.96224189 | 0.96283186 | 0.12321530 |
| closest_negative_distance=5 | 1111 | 0.96939694 | 0.96759676 | 0.08947002 |
| closest_negative_distance=6 | 86 | 0.98837209 | 1.00000000 | 0.03853809 |

| Control | Original Accuracy | Original NLL | Corrected Accuracy |
|---|---:|---:|---:|
| audio_absent | 0.25000000 | 1.38629436 | n/a |
| video_absent | 0.25000000 | 1.38629436 | n/a |
| audio_swap | 0.01771654 | 7.77989635 | 0.94825647 |
| video_swap | 0.01968504 | 7.75971358 | 0.94825647 |

Absent-input uniform25% is the explicit prototype policy, not a trained
unimodal result. Swap corrected labels restore matched metrics. All3556
custom-ID reverse-order decisions are exact for this policy.

Calibrated train-minus-development Accuracy gap: 4.08979438pp.
Prospective policy eligibility: False.


## hard candidates

| Variant / split | Accuracy | NLL | Brier | ECE | Mean confidence |
|---|---:|---:|---:|---:|---:|
| Parent / development | 0.84083240 | 0.41330416 | 0.22674230 | 0.01889695 | 0.82663369 |
| Candidate raw / train | 0.93517157 | 0.19016916 | 0.10250871 | 0.05338990 | 0.88178166 |
| Candidate raw / calibration | 0.79773869 | 0.50002881 | 0.27535056 | 0.03323122 | 0.82080822 |
| Candidate raw / validation | 0.84111361 | 0.39903600 | 0.22384125 | 0.01982412 | 0.85194702 |
| Candidate calibrated / train | 0.93517157 | 0.21116326 | 0.11053215 | 0.07528080 | 0.85989076 |
| Candidate calibrated / calibration | 0.79773869 | 0.49441532 | 0.27521803 | 0.03211489 | 0.79309378 |
| Candidate calibrated / validation | 0.84111361 | 0.39825375 | 0.22413430 | 0.01860703 | 0.82656422 |

Paired candidate-minus-parent intervals: whole development speaker groups,
10,000 resamples, seed17, linear percentile95%. Nine groups; quartet roles
are correlated. This observed development comparison is not a blind test.

| Metric | Difference | Lower | Upper |
|---|---:|---:|---:|
| accuracy | 0.00028121 | -0.01000000 | 0.01005747 |
| nll | -0.01505041 | -0.03342007 | 0.00345120 |
| brier | -0.00260800 | -0.01199213 | 0.00696700 |
| mean_confidence | -0.00006947 | -0.00321792 | 0.00291701 |
| ece | -0.00028992 | -0.01347974 | 0.00744840 |

| Stratum | Roles | Parent Acc | Candidate Acc | Candidate NLL |
|---|---:|---:|---:|---:|
| condition=l | 1800 | 0.84333333 | 0.85055556 | 0.36644859 |
| condition=p | 1756 | 0.83826879 | 0.83143508 | 0.43085584 |
| closest_negative_distance=1 | 86 | 0.56976744 | 0.52325581 | 0.97883829 |
| closest_negative_distance=2 | 812 | 0.73275862 | 0.72783251 | 0.62407437 |
| closest_negative_distance=3 | 2308 | 0.87391681 | 0.87694974 | 0.33002087 |
| closest_negative_distance=4 | 350 | 0.94000000 | 0.94571429 | 0.18163908 |

| Control | Original Accuracy | Original NLL | Corrected Accuracy |
|---|---:|---:|---:|
| audio_absent | 0.25000000 | 1.38629436 | n/a |
| video_absent | 0.25000000 | 1.38629436 | n/a |
| audio_swap | 0.05455568 | 5.61138794 | 0.84111361 |
| video_swap | 0.05174353 | 5.64542453 | 0.84111361 |

Absent-input uniform25% is the explicit prototype policy, not a trained
unimodal result. Swap corrected labels restore matched metrics. All3556
custom-ID reverse-order decisions are exact for this policy.

Calibrated train-minus-development Accuracy gap: 9.40579578pp.
Prospective policy eligibility: True.

## Decision and limitations

`CLOSE_SIX_COEFFICIENT_COMPARISON_KEEP_PARENT`.

Both policies must meet the frozen gate. Random Accuracy and Brier worsen,
and its NLL improvement is below1%; the comparison is closed, parent retained.
Hard Accuracy gains only one correlated role; its paired Accuracy and NLL
intervals include zero. Distance1 hard stratum has86 roles and loses4/86
correct answers; it is reported despite falling below the100-role gate.
Training NLL descent does not establish development improvement or a model
capacity ceiling. No extra steps/seeds, per-policy fits, coefficient search
or runtime/product promotion are justified by this result.

Existing native runtime stays on equal weights. Four-modality95–98%,
independent task/OOD quality and full resident integration remain unmet.

## Actual execution and artifacts

- primary: `C:\CodexArtifacts\pretrained-reuse-r2\lombard-slot-weights-01`, `PASS_FIXED_SLOT_WEIGHT_COMPARISON`;
  20.450139s, peak CPU 546365440 bytes;
  `2026-10-15T03:07:54.801180+00:00` to `2026-10-15T03:08:15.216271+00:00`;
  command: `python -m scripts.train_lombard_correspondence_weights --output C:/CodexArtifacts/pretrained-reuse-r2/lombard-slot-weights-01`.
- replay: `C:\CodexArtifacts\pretrained-reuse-r2\lombard-slot-weights-replay-01`, `PASS_FIXED_SLOT_WEIGHT_COMPARISON`;
  26.115426s, peak CPU 546881536 bytes;
  `2026-10-15T03:08:29.856335+00:00` to `2026-10-15T03:08:55.937639+00:00`;
  command: `python -m scripts.train_lombard_correspondence_weights --output C:/CodexArtifacts/pretrained-reuse-r2/lombard-slot-weights-replay-01 --replay-reference C:/CodexArtifacts/pretrained-reuse-r2/lombard-slot-weights-01 --replay-sha256 faf392060bfc8608ec92be4b85b025e33f42a4275b207b44baa960932ab27703`.
- smoke: `C:\CodexArtifacts\pretrained-reuse-r2\lombard-slot-weights-smoke-01`, `PASS_TRAIN_ONLY_SLOT_WEIGHT_SMOKE`;
  17.645516s, peak CPU 547561472 bytes;
  `2026-10-15T03:04:07.250298+00:00` to `2026-10-15T03:04:24.863107+00:00`;
  command: `python -m scripts.train_lombard_correspondence_weights --smoke --output C:/CodexArtifacts/pretrained-reuse-r2/lombard-slot-weights-smoke-01`.

Full primary/replay weights, every iteration, raw/calibrated metrics,
per-example IDs/targets/probabilities/log probabilities, controls and strata
are retained outside Git. Small JSON includes all attempted run summaries,
payload hashes, effective protocol/source revisions and uncertainty results.
Terminal publication receipt checks input and source hashes before/after.
CPU/NoTorch/Ruff/manifests and exact pushed-commit CI are recorded separately.

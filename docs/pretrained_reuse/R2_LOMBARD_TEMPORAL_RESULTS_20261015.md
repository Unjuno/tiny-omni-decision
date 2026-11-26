# Native temporal readout: completed comparison and limits

All five fresh CPU fits completed24epochs/768updates and passed separate-process
best/final train+validation replay, calibration, saved metrics/sampling/optimizer
and frozen-source checks. This is a six-slot speech-command task, not general
Omni quality. All51options, including25letters, and4080/398/889 grouped utterances
are retained. Nine development speakers and checkpoint selection limit inference.

## Result and decision

Retaining chronological visual information improves video-only macro Accuracy
29.85%→48.01%, NLL1.734875→1.377506 and Brier .764713→.609485.
Paired whole-speaker95% Accuracy difference is+18.17pp[13.79,21.47].
ECE worsens .027221→.043730; the weakest slot(letter) remains6.97%.
This changes temporal variation as well as order; not a pure order-only effect.

Joint broadcast/ordered reach99.53%/99.51% macro Accuracy, minimum slot98.20%
and complete-command97.19%/97.08%. **Mismatching video changes0/5334 slot
answers in either arm**, with nearly unchanged NLL. The higher score versus
audio98.18% does not establish matched audiovisual semantic benefit. A learned
constant visual state, extra effective parameterization or optimization could
contribute; those explanations are not yet measured. Audio-only's absent visual
state is forced zero, so equal registered parameters do not imply equal active
capacity. Do not label this a successful joint Omni model or promote it.

Ordered-minus-broadcast joint Accuracy−.01875pp[−.05624,0], NLL interval
includes no change. More temporal readout capacity/budget is not justified by
this comparison alone. Keep all results; next diagnose fixed visual-state/mask
contributions before another fusion fit. Video-only quality remains inadequate.

## Selected calibrated checkpoints

| Arm | Best epoch | Accuracy | NLL | Brier | ECE | Min slot | All-six correct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| audio | 24 | 98.18% | 0.053902 | 0.026349 | 0.009388 | 94.49% | 89.76% |
| video_broadcast | 5 | 29.85% | 1.734875 | 0.764713 | 0.027221 | 5.17% | 0.00% |
| video_ordered | 18 | 48.01% | 1.377506 | 0.609485 | 0.043730 | 6.97% | 0.22% |
| joint_broadcast | 23 | 99.53% | 0.016100 | 0.007527 | 0.003492 | 98.20% | 97.19% |
| joint_ordered | 23 | 99.51% | 0.016036 | 0.007503 | 0.003078 | 98.20% | 97.08% |

## Per-slot calibrated metrics

| Arm | Slot | Count | Accuracy | NLL | Brier | ECE | Mean confidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| audio | command | 889 | 99.78% | 0.008690 | 0.004239 | 0.005445 | 0.993556 |
| audio | color | 889 | 99.55% | 0.018761 | 0.008707 | 0.009031 | 0.988424 |
| audio | preposition | 889 | 96.06% | 0.101987 | 0.050967 | 0.011583 | 0.970438 |
| audio | letter | 889 | 94.49% | 0.167356 | 0.081225 | 0.018706 | 0.934099 |
| audio | number | 889 | 99.55% | 0.017996 | 0.008396 | 0.006257 | 0.989244 |
| audio | adverb | 889 | 99.66% | 0.008623 | 0.004560 | 0.005308 | 0.995527 |
| video_broadcast | command | 889 | 43.19% | 1.198851 | 0.660841 | 0.034101 | 0.455923 |
| video_broadcast | color | 889 | 31.72% | 1.363242 | 0.738530 | 0.014516 | 0.302694 |
| video_broadcast | preposition | 889 | 30.82% | 1.353563 | 0.733836 | 0.031523 | 0.289875 |
| video_broadcast | letter | 889 | 5.17% | 3.176061 | 0.956598 | 0.004796 | 0.056539 |
| video_broadcast | number | 889 | 17.21% | 2.194040 | 0.879015 | 0.016267 | 0.159511 |
| video_broadcast | adverb | 889 | 50.96% | 1.123492 | 0.619460 | 0.062126 | 0.448416 |
| video_ordered | command | 889 | 61.64% | 0.861149 | 0.475114 | 0.049341 | 0.657562 |
| video_ordered | color | 889 | 38.36% | 1.277091 | 0.698046 | 0.038112 | 0.398627 |
| video_ordered | preposition | 889 | 51.63% | 1.043318 | 0.580989 | 0.024577 | 0.527939 |
| video_ordered | letter | 889 | 6.97% | 3.167307 | 0.956179 | 0.010921 | 0.067881 |
| video_ordered | number | 889 | 47.92% | 1.419426 | 0.673963 | 0.103613 | 0.375577 |
| video_ordered | adverb | 889 | 81.55% | 0.496745 | 0.272619 | 0.035818 | 0.781728 |
| joint_broadcast | command | 889 | 100.00% | 0.002076 | 0.000655 | 0.001842 | 0.998158 |
| joint_broadcast | color | 889 | 99.89% | 0.007941 | 0.002675 | 0.001791 | 0.998798 |
| joint_broadcast | preposition | 889 | 99.10% | 0.033976 | 0.016420 | 0.007304 | 0.986327 |
| joint_broadcast | letter | 889 | 98.20% | 0.050844 | 0.024802 | 0.008440 | 0.982563 |
| joint_broadcast | number | 889 | 100.00% | 0.001737 | 0.000609 | 0.001548 | 0.998452 |
| joint_broadcast | adverb | 889 | 100.00% | 0.000027 | 0.000001 | 0.000027 | 0.999973 |
| joint_ordered | command | 889 | 100.00% | 0.002045 | 0.000647 | 0.001811 | 0.998189 |
| joint_ordered | color | 889 | 99.89% | 0.007986 | 0.002603 | 0.001729 | 0.998873 |
| joint_ordered | preposition | 889 | 98.99% | 0.033997 | 0.016490 | 0.005046 | 0.986352 |
| joint_ordered | letter | 889 | 98.20% | 0.050281 | 0.024546 | 0.008210 | 0.982745 |
| joint_ordered | number | 889 | 100.00% | 0.001893 | 0.000731 | 0.001661 | 0.998339 |
| joint_ordered | adverb | 889 | 100.00% | 0.000014 | 0.000000 | 0.000014 | 0.999986 |

## Fixed selected controls

| Arm | Control | Accuracy | NLL | Brier | ECE |
| --- | --- | --- | --- | --- | --- |
| joint_broadcast | matched | 99.53% | 0.016100 | 0.007527 | 0.003492 |
| joint_broadcast | reverse | 99.53% | 0.016100 | 0.007527 | 0.003492 |
| joint_broadcast | audio_mismatch | 31.46% | 14.260354 | 1.361654 | 0.680458 |
| joint_broadcast | video_mismatch | 99.53% | 0.016082 | 0.007517 | 0.003680 |
| joint_broadcast | audio_missing | 18.95% | 1.946875 | 0.832560 | 0.076651 |
| joint_broadcast | video_missing | 97.34% | 0.093656 | 0.043870 | 0.026059 |
| joint_ordered | matched | 99.51% | 0.016036 | 0.007503 | 0.003078 |
| joint_ordered | reverse | 99.18% | 0.026852 | 0.012106 | 0.008584 |
| joint_ordered | audio_mismatch | 31.44% | 14.220596 | 1.361744 | 0.680640 |
| joint_ordered | video_mismatch | 99.51% | 0.016036 | 0.007503 | 0.003078 |
| joint_ordered | audio_missing | 18.86% | 1.944125 | 0.832132 | 0.076635 |
| joint_ordered | video_missing | 97.34% | 0.092399 | 0.043230 | 0.026933 |

Reversal means16cached visual tokens, not raw-frame re-encoding. Broadcast
is reversal-invariant by construction. Ordered reversal changes34slot answers
and worsens Accuracy/NLL; this alone does not prove matching video semantics.
Donor controls retain p/l condition and exact donor marginal with different speakers.
Absent modality uses unseen masks(OOD); poor scores do not quantify natural missingness.
Audio mismatch is highly confident and wrong: calibration does not transfer to this
artificial contradictory-input control. No controls were used for model selection.

## Learning curves, selection and consumption

All120epoch observations, train/validation per-slot Accuracy/NLL/Brier/ECE are in
[curve CSV](R2_LOMBARD_TEMPORAL_CURVES_20261015.csv), with raw/final and grouped
intervals in [complete small JSON](R2_LOMBARD_TEMPORAL_RESULTS_20261015.json).
Earliest minimum unrounded raw validation macro NLL chooses best; final24is
saved separately. Calibration uses only398independent rows and the fixed17-point
temperature grid. Same106355head, seed17initialization/sample orders, batch128,
fixed AdamW LR/WD .001, no encoder training/noise/source/frame changes.

| Arm | Selected consumed assets | Total assets | Unique assets | Raw best NLL | Raw final NLL | Final train Acc | Final val Acc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| audio | 97920 | 97920 | 4080 | 0.062119 | 0.062119 | 100.00% | 98.18% |
| video_broadcast | 20400 | 97920 | 4080 | 1.749506 | 2.112193 | 54.00% | 31.31% |
| video_ordered | 73440 | 97920 | 4080 | 1.414200 | 1.445742 | 68.85% | 51.22% |
| joint_broadcast | 93840 | 97920 | 4080 | 0.022695 | 0.023314 | 99.99% | 99.51% |
| joint_ordered | 93840 | 97920 | 4080 | 0.022618 | 0.023186 | 99.99% | 99.49% |

Each arm consumes587520slot decisions,24480unique decisions and93840repeated
asset presentations. Five arms together489600presentations share4080unique
assets; do not count five runs as unique data. The smoke adds40training presentations
over4unique assets/10updates. Its gradient-only timing probe separately adds384
presentations/0updates and changes no weights. It is not part of main-run consumption.
Video broadcast strongly overfits after best5; ordered improves through18then
train NLL continues falling while development NLL worsens, even as Accuracy rises.
Joint23→24has slight development NLL deterioration; audio still improves through24.
Do not automatically choose last or extend budgets. Frozen-source pretraining
overlap remains UNKNOWN. Old pooled results are preserved and are not an equal-capacity
causal comparison. Bootstrap10000/seed17whole-speaker/linear percentile95%,
nine groups, single seed and observed-development selection—not blind inference.

## Resources, reproducibility and preservation

Source commit `80290ee6c1158731476b5100165f6cb3a5e4afd0`. All frozen physical files/runtime pins
and complete training artifact hashes pass before/terminal rechecks. The smaller
result JSON records exact commands, source/split/cache/config/report hashes and times.
Artifacts remain outside Git under`C:/CodexArtifacts/pretrained-reuse-r2/`:
- `lombard-temporal-smoke-01`: 22.922s, peak 1.381GiB, receipt SHA`2676b88d2b0c44986986b457688ce24cf0b4812b32b9cc297df3331caa1b2f52`.
- `lombard-temporal-full-01`: 144.346s, peak 1.736GiB, receipt SHA`09ef033c461313a84d34a7db5b6a754c57619871ba1ea9e7f9bdb746e830ec0e`.
- `lombard-temporal-evaluation-01`: 26.151s, peak 1.807GiB, receipt SHA`c97d975df27f4931be0b256e5a0ed90f476b2ba7dbbc7534085b7667f3694fdc`.

Intel i7-12700H, CPU4threads/interOp1, FP32/deterministic, Python3.11.9,
Torch2.6.0+cu124, NumPy2.4.6. GPU unused/encoders never loaded in fitting; no
cloud cost. CPU tests overlapped part of fitting; wall time is shared-PC time.
7200s/3GiB/128MiB/1GiB reserve limits held. Existing GPU8GiB guard/interrupted
run, v1/v2Teacher weights/config/corpora, source caches and sealed data untouched.
Best/final weights, optimizer/global step/RNG/orders retained; no verified strict resume.
Separate evaluator keeps original requires_grad flags for numeric parity but runs
eval/no_grad without optimizer or updates and checks unchanged model tensors.

Native encoders122573568 plus106355head/2304normalization =122682227resident
parameters for this AV composition. Deployment encoder costs remain included;
not a packaged four-modality model/size/latency proof. JSON native-option reorder
checks pass all889selected observations per arm. Weights/media are not added to Git.

**Remaining:** genuine input-dependent AV benefit, natural-noise/task/OOD transfer,
all four modalities, fresh audit and complete <=200M deployment. The goal remains active.

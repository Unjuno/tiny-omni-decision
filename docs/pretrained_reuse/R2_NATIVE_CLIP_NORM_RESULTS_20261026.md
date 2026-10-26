# Native Decision gradient clipping comparison (2026-10-26)

## Outcome

One controlled seed-17 comparison completed: global gradient clipping at 2.0
against the preserved 1.0 run. The candidate reached all 32,768 updates and a
separate-process replay reproduced best and final outputs exactly. This is a
small validation gain at the selected checkpoint, accompanied by worse text
calibration and pronounced late overfitting. It does not justify product
promotion, a new training budget, or a claim that clipping alone solves the
multimodal quality gap.

## Frozen conditions and provenance

- Branch: `codex/native-clipnorm-comparison-20261026`.
- Candidate runner source: `533c3c505f402cd2fcf37c7fcbc9c4ccd99ba3a4`.
- Preserved clip-1.0 fit source: `8756a09f582554c0321ed5ece71846d6b19469cd`.
- Both use the same static config SHA-256:
  `19e4a1f005856684950fb229a12d757768eb7369787ad08b40b3f85de2ba3e45`.
- Candidate effective config SHA-256:
  `824722ecfeb8f8f71c1c7639ded4e3699c39d025295247960745f4b2ecb90865`.
  Its sole intentional difference is `training.clip_norm: 2.0` rather than
  `1.0`.
- Initialization: same original native seed-17 readout, SHA-256
  `3ce7f72450e16946b1fb113c3fa8793f1d6a59f2b6221564cff7b30feebfba1c`;
  no warm start.
- Sampling-order SHA-256 (both):
  `267c2144c4bb3045e1bbc2365c83618306a086a9014f3ac7e44ac1e19bcc11bf`.
- LR-plan SHA-256 (both):
  `14234584342acb62df7d3987fdc835ec7beec3b04f5070ae864a9eac0c6becc9`.
- Validation-ID SHA-256 (both):
  `db7d0f1d09319a48a7725ed36ccb515cdcc85a409f019400cf0062be38768f30`;
  24,346 fixed validation examples.
- No calibration or sealed-audit data loaded; no encoder updates; cloud cost
  was $0. Only local GPU compute was used.

The old norm-1.0 run remains untouched at
`C:/CodexArtifacts/pretrained-reuse-r2/native-long-budget-fit-01`. New candidate
artifacts are outside Git under `C:/CodexArtifacts/pretrained-reuse-r2/`:
`native-long-budget-clip2-smoke-01`, `native-long-budget-clip2-fit-01`, and
`native-long-budget-clip2-replay-01`.

## Execution and accounting

The 32-update smoke passed and its save/reload check was exact. It projected
3,582 seconds for a full run, below the two-hour preflight limit. The candidate
fit ran from 2026-10-26 20:33:39 to 21:17:52 JST and completed 32,768/32,768
updates without performance early stopping. Best checkpoint selection remained
validation-only and selected step 12,288; the control selected step 8,192.
Best and final checkpoints were retained separately.

Recorded runner invocations (Python 3.11.9, project root on `PYTHONPATH`):

```text
scripts/train_native_long_budget.py --phase smoke --policy repeat --clip-norm 2.0 --output C:/CodexArtifacts/pretrained-reuse-r2/native-long-budget-clip2-smoke-01
scripts/train_native_long_budget.py --phase train --policy repeat --clip-norm 2.0 --output C:/CodexArtifacts/pretrained-reuse-r2/native-long-budget-clip2-fit-01 --reference C:/CodexArtifacts/pretrained-reuse-r2/native-long-budget-clip2-smoke-01 --reference-sha256 dc82d41592d8ebe81902a2bfee0ddd60c7be5d5cfe4183ce069ea16218ad91db
scripts/train_native_long_budget.py --phase replay --policy repeat --clip-norm 2.0 --output C:/CodexArtifacts/pretrained-reuse-r2/native-long-budget-clip2-replay-01 --reference C:/CodexArtifacts/pretrained-reuse-r2/native-long-budget-clip2-fit-01 --reference-sha256 2e056cc6b0968e10ef2343c062b03e3b8baa95a848a97758fbc465291e3cf642
```

The sampler performed 262,144 example presentations (8 per update): 6,656
unique questions and 255,488 repeated presentations; 5,107 unique observations,
3,114 unique media assets, and 3,382 groups. The selected step-12,288 checkpoint
represents 98,304 presentations, with 6,656 unique questions and 91,648 repeats.
Thus the full-run and selected-checkpoint consumption are not interchangeable.
All four modalities had 65,536 presentations over the full run.

Candidate runtime was 2,652.995 s total (1,887.103 s training and 717.758 s
evaluation); peak GPU allocated/reserved memory was 209,224,704 / 327,155,712
bytes and peak host working set was 3,753,283,584 bytes. Device: NVIDIA GeForce
RTX 3080 Laptop GPU; Python 3.11.9, PyTorch 2.6.0+cu124, CUDA 12.4, NumPy 2.4.6,
float32. The preserved control took 2,550.882 s total and peaked at the same
GPU memory values.

## Learning curve

Macro metrics are an unweighted arithmetic mean across the four modalities.
Accuracy / NLL / Brier / ECE:

| Update | Clip 1.0 | Clip 2.0 |
|---:|---:|---:|
| 512 | .2906 / 1.7645 / .7546 / .0855 | .2924 / 1.7219 / .7443 / .0733 |
| 1,024 | .3049 / 1.6938 / .7414 / .0775 | .2937 / 1.7091 / .7461 / .0934 |
| 2,048 | .3283 / 1.7396 / .7478 / .1228 | .3253 / 1.7018 / .7420 / .1113 |
| 4,096 | .3453 / 1.6746 / .7270 / .1154 | .3569 / 1.6611 / .7221 / .1126 |
| 8,192 | .3742 / 1.6302 / .7057 / .1261 | .3611 / 1.6301 / .7131 / .1297 |
| 12,288 | .3998 / 1.6952 / .7087 / .1502 | **.3875 / 1.6145 / .6965 / .1270** |
| 16,384 | .4095 / 2.0496 / .7509 / .2050 | .4113 / 1.7348 / .7187 / .1737 |
| 24,576 | .4461 / 3.6151 / .7985 / .2838 | .4505 / 2.6529 / .7812 / .2811 |
| 32,768 | .4570 / 4.8481 / .8155 / .3116 | .4673 / 3.5272 / .8102 / .3243 |

The fixed validation set improves in accuracy late, but its NLL, Brier and ECE
degrade sharply after the selected region. Fixed train-probe accuracy keeps
rising while validation calibration worsens (clip 2.0: at step 12,288 train
probe .5234 accuracy / 1.2036 NLL versus validation .3875 / 1.6145; at 32,768
train probe .8284 / .4668 versus validation .4673 / 3.5272). These are fixed
probe-versus-validation comparisons, not a rolling-window generalization gap.
They indicate pronounced late overfitting and overconfidence.

## Selected checkpoint, same validation IDs

All rows show Accuracy / NLL / Brier / ECE.

| Metric | Clip 1.0, step 8,192 | Clip 2.0, step 12,288 |
|---|---:|---:|
| Macro across modalities | .3742 / 1.6302 / .7057 / .1261 | .3875 / 1.6145 / .6965 / .1270 |
| Audio (n=11,166) | .6246 / 1.0706 / .4655 / .0769 | .6333 / .9994 / .4477 / .0587 |
| Image (n=2,048) | .3540 / 1.7989 / .7353 / .0766 | .3828 / 1.6523 / .6954 / .0789 |
| Text (n=10,732) | .2160 / 2.0022 / .8378 / .0802 | .2192 / 2.2037 / .8571 / .1160 |
| Video (n=400) | .3375 / 1.5017 / .7556 / .1898 | .3450 / 1.5064 / .7615 / .1850 |

The selected candidate raises macro Accuracy by 1.33 percentage points, with
modestly lower macro NLL and Brier but slightly higher ECE. Text Accuracy is
nearly flat while its NLL, Brier and ECE worsen; that trade-off rules out calling
the result an across-the-board improvement. Video's validation sample is only
400 examples, so its 0.75-point change is weak evidence.

At the final step, sample-weighted accuracy is 46.73% versus 45.70% for control,
but macro NLL/ECE are 3.5272/.3243 versus 4.8481/.3116. Late accuracy gains do
not imply calibrated quality. There is no inference here that the model family
has reached a ceiling.

## Interpretation and limits

This single-seed result provides limited evidence that a 2.0 clip threshold can
improve the selected checkpoint on this exposed fixed validation set. It does
not isolate gradient clipping as the only cause of downstream quality, establish
generalization, or demonstrate the roadmap's 95–98% goal. The late training
curve shows clear overfitting. Do not extend the budget or promote this candidate
based on this run. Preserve the clip-1.0 baseline and continue R2 task/data
quality work under the roadmap's predeclared gates.

No sealed audit was opened or re-evaluated. No prior v1/v2 run, corpus, config,
checkpoint, or artifact was overwritten. The only non-code changes from this
comparison are this report and the roadmap note.

## Artifact integrity

| Artifact | SHA-256 |
|---|---|
| Control `result.json` | `645bedbe73ed508fad68c4f636a41df6c8630cd996641c44ce2a87d31550250b` |
| Candidate `result.json` | `2e056cc6b0968e10ef2343c062b03e3b8baa95a848a97758fbc465291e3cf642` |
| Candidate effective config | `824722ecfeb8f8f71c1c7639ded4e3699c39d025295247960745f4b2ecb90865` |
| Candidate best checkpoint | `287874ed88f36eb5cbbd0e90a3f181b3898bec4875a2a5b9ca2207ae230fd0c3` |
| Candidate final checkpoint | `487e267e9e6999245df6f4b965dc5fab756671d08c4c9c27db4c94ae3927aa50` |
| Candidate learning curve | `3e1b57b6efd1675677a4fb45c70c99a6d063278aa15b1dc8c609110eeda6277f` |
| Candidate training history | `019181458b8bb37d3c38c8d51c1331a9e33519d9663d7baae70db660484e0629` |
| Candidate replay receipt | `e7ecfd7da049b88054743733750e16b2f32dce550c818d07da6feab76eeac22c` |

All listed run-artifact hashes were checked against their external files; the
candidate replay has `replay_max_abs = 0.0` for best and final outputs. The full
predictions and task/source stratification remain in the external learning
curve; model weights and media are intentionally excluded from Git.

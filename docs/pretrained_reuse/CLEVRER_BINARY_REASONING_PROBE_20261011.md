# CLEVRER binary reasoning probe (2026-10-11)

This one-time Path A readout asks whether the already cached VideoPrism scene
representation plus frozen MiniLM question/candidate features support
CLEVRER's non-descriptive reasoning labels. Each upstream multiple-choice
candidate is converted to a separate binary Decision: the input contains the
original question and candidate statement, options are `[wrong, correct]`,
and the target is the upstream choice label. The frozen 8-frame VideoPrism
features and MiniLM tower are not updated; only the small shared candidate
scorer is trained. This is a development probe, not a blind or product
benchmark.

## Data and integrity

| Item | Result |
|---|---|
| Source/split | `MIT-IBM/CLEVRER@98b842082ba4f7c18b6b9e3f39145871782a65ef`, CC0-1.0; 256 train scenes and 200 validation scenes; 456 video assets. Scene overlap and video SHA-256 overlap are 0. These validation scenes were previously used to select the descriptive VideoPrism readout, so this is not independent of prior development. |
| Task size | Train: 3,794 binary choices (explanatory 1,560; predictive 352; counterfactual 1,882) from 1,098 parent questions. Validation before content filtering: 3,043; 29 rows removed for 27 exact normalized question-plus-options collisions with train. Final validation: 3,014 (explanatory 1,234; predictive 255; counterfactual 1,525) from 868 parent questions. Exact normalized content overlap after filtering is 0. |
| Raw question hashes | Train `11181da673d223f41fb596aacfbbd3ff83d39af7f09a3210549e98cb283714b4`; validation `fdf841678a476655b906165e6ee4b0ed1516785c5ab927c8d0a4614e4e27e10d`. |
| Sample manifest hashes | Train `5cd4e5155b0161e35b10aad2c98e14afba7d46407e635bd1c08d786e1e51765d`; validation `53aa6f01447a43337a601279daf620c0df91d807773522cc92230fcab173db7c`; fetch report `75f0511bb94df2abd7e2734c0881af0282ed3797ec7b725cc07fa7dc8fd84a44`. |
| Frozen features | VideoPrism-B, 8 frames, 768 pooled features; cache SHA-256 `40eaff323e3654bf34653950d5fea4cd532f1a0e096438ec9cb73703dc589ea1`; pinned checkpoint SHA-256 `61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29`. MiniLM `sentence-transformers/paraphrase-MiniLM-L3-v2@4ca70771034acceecb2e72475f72050fcdde4ddc`, weights SHA-256 `cf1e4e2d420c664973037c3c73125d7a8fc69952495093ef8f50596f8943a433`. Both encoders were frozen. |
| Leakage/audit | Validation exact-content collisions were excluded; scene/media overlap is 0. The sealed audit was not loaded. The validation assets were used for earlier descriptive readout selection, and the validation questions were used for selecting this checkpoint; results are development-only. |

### Parent-question target structure audit

The source labels were checked on the saved normalized examples before
interpreting the binary-choice metrics. They do not define one winning choice
per parent question: a parent can have zero, one, or several choices labeled
`correct`. On the 868 validation parent questions (3,014 statement-level
examples), the number of `correct` statements was 0 for 68 questions, 1 for
337, 2 for 229, 3 for 218, and 4 for 16. By task, the corresponding counts
were:

| Task | Parent questions with 0 / 1 / 2 / 3 / 4 correct statements |
|---|---|
| Explanatory | 57 / 74 / 101 / 92 / 16 (340 total) |
| Predictive | 10 / 128 / 0 / 0 / 0 (138 total) |
| Counterfactual | 1 / 135 / 128 / 126 / 0 (390 total) |

The 1,098 train parent questions have counts 63 / 454 / 316 / 235 / 30.
The separate fresh-scene control validation also has a multi-label structure:
among 894 parents, counts are 84 / 413 / 234 / 154 / 9 (0 through 4 correct
statements). The control rows are the same candidates used for that already
reported development comparison; no new prediction or model selection was run.

Therefore, the current conversion—one `[wrong, correct]` decision per
candidate statement—is faithful to the source labels. Its reported Accuracy,
NLL, Brier and ECE are statement-level binary metrics. They must not be
reinterpreted as original-question single-winner accuracy, and probabilities
must not be renormalized across all statements under a single-target
assumption. This label structure helps explain why a single multiclass
choice-probability benchmark cannot be derived from these rows without
changing the task. It does not by itself explain the near-chance binary
validation result.

Audited artifacts, stored outside Git:

- Train normalized decisions SHA-256
  `b98c3089f9850fa2343c77561a963306fd0b8201912e38000143fc72da36f032`.
- Original validation normalized decisions SHA-256
  `57e663bef75c4d652ab71083476b39f3b9d3a0e5913a4e8f6e0cad0245e9c587`.
- Fresh-scene control validation source SHA-256
  `1eaf8b65eb632f930b201bd9e4d48b904d69aaa49f85764298083740d6bd5ae0`.
- Fresh-scene control predictions SHA-256
  `251fce980966dae57733b378934e36df4712e446667b1603280898d0aeeeaffd`.

## Training and results

The seed-17 shared candidate MLP has hidden size 128 and 295,169 trainable
parameters. It was trained with AdamW (learning rate 0.001, weight decay
0.001), 24 epochs, batch size 64 and 1,440 optimizer updates. Each epoch
presented all 3,794 unique training choices once; across the run this was
91,056 presentations, 3,794 unique examples and 87,262 repeated presentations.
Loss weights give equal aggregate weight to each question type and then equal
weight to each scene within type. The checkpoint selector was minimum
validation macro question-type NLL; epoch 24 was selected. No other experiment
or seed was run.

| Validation group | n / scenes | Accuracy | NLL | Brier | ECE-15 |
|---|---:|---:|---:|---:|---:|
| Macro across question types | 3,014 / — | 0.5194 | 0.6918 | 0.4987 | 0.0170 |
| Example-weighted | 3,014 / 200 | 0.5179 | 0.6921 | 0.4989 | 0.0064 |
| Explanatory | 1,234 / 194 | 0.5429 | 0.6902 | 0.4971 | 0.0181 |
| Predictive | 255 / 138 | 0.5176 | 0.6916 | 0.4985 | 0.0068 |
| Counterfactual | 1,525 / 200 | 0.4977 | 0.6936 | 0.5005 | 0.0261 |

Ten thousand scene-cluster bootstrap replicates (seed 17; all questions for
each sampled scene retained) give macro Accuracy 95% percentile interval
`[0.4953, 0.5439]` and macro NLL interval `[0.6890, 0.6947]`. Both include
chance-level behavior (0.5 Accuracy / approximately 0.693 binary NLL). Per-task
validation label-majority Accuracy is 0.501 explanatory, 0.502 predictive,
and 0.504 counterfactual. This probe does not establish above-chance general
CLEVRER reasoning.

| Epoch | Updates | Train macro Accuracy | Validation macro Accuracy | Train macro NLL | Validation macro NLL |
|---:|---:|---:|---:|---:|---:|
| 1 | 60 | 0.5034 | 0.5018 | 0.6966 | 0.6969 |
| 4 | 240 | 0.5357 | 0.5036 | 0.6930 | 0.6931 |
| 8 | 480 | 0.5429 | 0.5093 | 0.6926 | 0.6929 |
| 12 | 720 | 0.5335 | 0.5111 | 0.6924 | 0.6929 |
| 16 | 960 | 0.5319 | 0.5149 | 0.6912 | 0.6925 |
| 20 | 1,200 | 0.5194 | 0.5060 | 0.6906 | 0.6935 |
| 24 (selected) | 1,440 | 0.5459 | 0.5194 | 0.6873 | 0.6918 |

Validation NLL fluctuated and ended at its best observed value; train-validation
macro gaps at the selected checkpoint are +0.0264 Accuracy and -0.0046 NLL
(train minus validation). This curve and near-chance result do not diagnose a
capacity or representation ceiling. No matched text-only or video-shuffled
control was run, so this result cannot establish incremental value from video.

## Runtime and artifacts

| Item | Result |
|---|---|
| Hardware/runtime | RTX 3080 Laptop GPU, 16 GiB; Python 3.11.9, PyTorch 2.6.0+cu124 / CUDA 12.4, Transformers 5.6.2, NumPy 2.4.6. |
| Wall time and memory | Input preflight 3.05 s; MiniLM feature extraction 2.74 s; readout training 16.14 s; evaluation 5.83 s; peak CUDA allocation 130,593,792 bytes. No paid/cloud compute. GPU returned idle. |
| Output directories | `C:\CodexArtifacts\pretrained-reuse-r2\clevrer-binary-reasoning-preflight-v0` and `C:\CodexArtifacts\pretrained-reuse-r2\clevrer-binary-reasoning-v0`; each was new. The run contains effective config, training history, validation decisions, per-example validation predictions/probabilities, selected checkpoint and report. |
| Hashes | Preflight report `e1c8e4d5b95c607461d4539316619767b66b9fd9587d2dc4ba3f238a32bfd196`; run report `af755a4637d3bfea818e83e36fa505410af30e5e1b0dda4796e3e9312b4b630a`; selected readout `c15f9184794717e5942ed45a2eec18e93b12ad9a15d4b27f8b3cdf4ff480be77`. Source commit `c5b9d25`; training script SHA-256 `fce93d91efd99973988407f2ac9a23c8604b2feb5fef499aa1f99b2e287f20f6`; config SHA-256 `6b545100b694ab75c16d5ecc62c25adb05140f6c0c0fe2ebb9025ea94a6d3cf9`. |

Reproduction from the repository root in PowerShell, using the already
downloaded pinned inputs and a new output directory only:

```powershell
$env:PYTHONPATH = 'src'
py -3.11 -m scripts.train_frozen_clevrer_binary_reasoning --preflight-only
py -3.11 -m scripts.train_frozen_clevrer_binary_reasoning
```

The config, corpus, source questions, VideoPrism cache/checkpoint and prior
artifacts were read-only. The follow-up text-only and video-shuffled controls
on content-disjoint scenes are documented in
[`CLEVRER_VIDEO_TEXT_CONTROLS_20261011.md`](CLEVRER_VIDEO_TEXT_CONTROLS_20261011.md).
The parent-question label structure is audited above. These results do not
justify shared-Lens training or model merging; further readout tuning on these
already-used development validations is not warranted. A stronger next
quality experiment needs a new, task-matched evaluation source or a new
predeclared task whose labels support the intended output semantics.

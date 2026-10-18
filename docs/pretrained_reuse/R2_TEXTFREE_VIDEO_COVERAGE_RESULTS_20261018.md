# Text-free CLEVRER scene coverage and temporal-readout comparison

## Outcome and correction to the initial experiment plan

Two readouts (mean-MLP and BiLSTM) completed 2,048 updates each on a 512-scene
train split. The initial run alone could not test a train-coverage effect: it
contained no smaller-data arm. The initial config's wording about doubling
coverage is retained in the hashed run record, but that run is not cited as a
coverage comparison.

To make that contrast valid, a deterministic 256-scene control was selected
from the 512-scene train pool, retaining both questions per scene. The control
uses the same 512 validation questions, same ordered IDs, same initial model
weights for each readout, same 2,048 updates, same 131,072 example
presentations, and validation every 16 updates. Both runs are development-only
and select checkpoints on that shared validation set.

Result: 512 scenes improve static descriptive performance for both readouts.
Temporal descriptive changes are uncertain and small. Macro validation NLL
improves for mean-MLP; BiLSTM's macro NLL interval includes zero. Both readouts
overfit sharply late in training. This is evidence that unique video coverage
helps static CLEVRER queries in this specific frozen VideoPrism/text-free
readout path; it does not establish a temporal reasoning gain, a general
video-model ceiling, or product readiness.

## Data, grouping, and integrity

- Dataset: CC0 `MIT-IBM/CLEVRER` revision
  `98b842082ba4f7c18b6b9e3f39145871782a65ef`.
- 512-scene arm: 1,024 train questions / 512 scenes / 512 media assets; 512
  validation questions / 256 distinct scenes / 256 media assets.
- Matched control: 512 train questions / 256 scenes / 256 media assets; the
  same 512 validation questions in exactly the same order. Scene, media and
  example IDs do not overlap across train and validation.
- The 256 scenes were ranked by `SHA256("17:" + scene_group_id)` and the first
  256 were retained; all rows for each selected scene remain together. The
  selected scene-list SHA-256 is
  `fae11faf3cac0b2a88bf8ac05e7ca964cc2f6b62cfc0ba8663d5d81f8fa48908`.
- There are repeated normalized question templates across distinct videos.
  The 512-scene training split shares 134 normalized question/option templates
  with validation (324 validation rows); the 256-scene control shares 99
  templates (278 validation rows). This is explicitly counted. Validation is
  not template-independent.
- All 768 selected video files were checked against their row media SHA-256.
  Full upstream archive SHA-256 verification was not recorded by the source
  acquisition receipt. The sealed audit and official test were not loaded.
- VideoPrism base f16r288 is frozen: 114,365,184 encoder parameters. Inputs are
  eight endpoint-inclusive frames; spatial tokens are pooled separately per
  frame. The only trainable weights are readout parameters: 226,865 for
  mean-MLP, 227,377 for BiLSTM. No language text is passed to either readout.

## Frozen execution and artifacts

The feature extraction used local CPU only: JAX 0.6.2 / Flax 0.11.2,
TensorFlow 2.19.1 and PyAV 18.1.0. It processed 768 unique videos in
1,043.64 seconds, with a recorded peak working set of 4,162,019,328 bytes.
The `[768, 8, 768]` float32 feature cache is 17,612,460 bytes, SHA-256
`851c2d1361fd409f25567123219b2c2278994c85517ab492252d98e53512857e`.

Readout fits used Python 3.11.9, PyTorch 2.6.0+cu124 and NumPy 2.4.6 on
Windows 10 build 26300, Intel Family 6 Model 154. CUDA was masked and the
selected device was CPU; peak allocated GPU memory was 0. Orbitune processes
were left alone. Cloud compute/cost: none / $0.

Both 512-scene arms were run under source commit
`adc62bf77d9c84883ad69beab048ec5a6c29ebcc`, config SHA-256
`c427639be852c5fdf1e0a329a66cb3633b949698de7f874619b09a4394c5687c`.
Their input sample hashes are train `2691c4494141c9685805977fdb42690269952061f7924cb90ccac3d7580ee62b`,
validation `322859a0e8885e228daff1f463aa5a2f14bf8d873aa54f8bea972926c6a58d34`,
fetch receipt `38e67ad1b39cedeba86333a1ea7114d2d44ae4c8dc183bec8bc53b7a3658ff52`,
and media/split receipt `59186ac0c608a4e578b9d89cad3fff7d9e8ab386d12dfeb35408896ac480e3be`.

The 256-scene control used source commit `05dddd6f53b539f02a802c8bf1842d58f91aad73`,
config SHA-256 `b4c2c645774a5191cad1d090502a888f626c81703585f61f04c872598ae45ddb`,
derived train manifest SHA-256 `44c2442770a839390f58f3df1752d39cbde0cde557788d654e2adac428f2ffb0`,
validation JSONL SHA-256 `8ab8f75fdc88adc87d33f8ac730e7483aad4d4d7f664305ca563516204707d6e`,
fetch receipt SHA-256 `434660e199f300243b411f1c43be3be769884de2afa17643332d6af5a66b41a7`,
and media/split receipt SHA-256 `1d51b16c093af2bea956e348bb11343d59c9d0456aa84f678528548529ff3916`.
Its ordered validation sample IDs match the 512-scene arm exactly; both ID-order
hashes are `9f3868d727568904ca55dac48aff6bc3c0dd656853dce1c87bd08f303f396804`.

Output directories (weights and raw predictions are intentionally outside Git):

- 512 scenes:
  `C:/CodexArtifacts/pretrained-reuse-r2/textfree-video-coverage-512train-256val-v1-attempt-01/fit-attempt-02`
- 256 scenes:
  `C:/CodexArtifacts/pretrained-reuse-r2/clevrer-coverage-256train-256val-textfree-control-v1/fit-attempt-01`
- Combined checkpoint/prediction hash ledger:
  `C:/CodexArtifacts/pretrained-reuse-r2/textfree-video-coverage-512train-256val-v1-attempt-01/artifact-hashes.json`
  (SHA-256 `8a8872541c4f551afcab0ef79f90546b60941789bde81a7967408337f834a9da`).

The 512-scene report SHA-256 is
`1be45aabb46530f1a126dceae5d1cf3708ba845311d5cc726dc8828b7b1ee23f`; the
256-scene report SHA-256 is
`cf903ccf90e156e40dbe383a64317beb8054cf0b0e87496582202827668d6f31`; the
paired coverage comparison SHA-256 is
`1e94b91e57dd5ab7d590c506ca0cbbafad81566120aebbbba8e3a1e9e8693b3e` for
per-question-type contrasts and
`5fb41fa9dfff71d53e4b2d09e21d6f464e6195878f7bfded97f8cd2d4b25010b` for
macro scene-cluster bootstrap contrasts. The complete ledger includes best and
final weights, histories, input preflight, selected/final train and validation
predictions, controls and run reports. Best checkpoint reload was bitwise
identical in both runs.

After the fit, all four selected checkpoints were loaded in a fresh Python
process and their selected-validation JSONL was regenerated. Every byte
matched the saved predictions for the corresponding readout/split, confirming
exact selected-checkpoint replay independent of the training process.
Independently reconstructing the seed-17 initial state also produced identical
state-dict hashes for each readout in both data arms: mean-MLP
`1e1236e74f42f7efce009388c9e8fe148ef4488748fba48926ceadfb56c7ba81` and
BiLSTM `d82967cb32bf95c731dbce730f91ed4783a804e17298f8bd0c9e80dff945caef`.

## Selected checkpoint results

Selection minimized mean NLL over static and temporal descriptive validation
questions, with the earliest checkpoint winning ties. Each row below reports
Accuracy / NLL / Brier / ECE (15 bins); `n=256` per question type.

| Train scenes / readout | Best step (selected exposures) | Static descriptive | Temporal descriptive | Macro across types |
|---|---:|---:|---:|---:|
| 256 / mean-MLP | 64 (4,096) | 50.00% / 1.0621 / .5957 / .1080 | 31.25% / 1.3426 / .7072 / .0771 | 40.63% / 1.2024 / .6515 / .0925 |
| 512 / mean-MLP | 112 (7,168) | 50.78% / 1.0404 / .5817 / .0786 | 28.91% / 1.3264 / .7003 / .0940 | 39.84% / 1.1834 / .6410 / .0863 |
| 256 / BiLSTM | 96 (6,144) | 50.39% / 1.0363 / .5834 / .0832 | 30.86% / 1.3537 / .7147 / .1388 | 40.63% / 1.1950 / .6490 / .1110 |
| 512 / BiLSTM | 176 (11,264) | 55.86% / 1.0010 / .5569 / .1140 | 32.03% / 1.3542 / .7124 / .1185 | 43.95% / 1.1776 / .6347 / .1162 |

The saved paired scene-cluster bootstrap used 5,000 resamples, seed 17, and
percentile 95% intervals. Differences are 512 scenes minus 256 scenes:

| Readout / question type | Accuracy difference (95% CI) | NLL difference (95% CI) | Brier difference (95% CI) |
|---|---:|---:|---:|
| mean-MLP / static | +0.78 pp [-5.08, +6.64] | -.0217 [-.0399, -.0036] | -.0140 [-.0275, -.0007] |
| mean-MLP / temporal | -2.34 pp [-8.98, +4.30] | -.0162 [-.0410, +.0075] | -.0070 [-.0211, +.0060] |
| BiLSTM / static | +5.47 pp [+0.39, +10.55] | -.0353 [-.0676, -.0037] | -.0265 [-.0458, -.0075] |
| BiLSTM / temporal | +1.17 pp [-4.69, +7.03] | +.0006 [-.0391, +.0433] | -.0023 [-.0234, +.0199] |

Macro, per-scene paired bootstrap contrasts were:

| Readout | Accuracy difference (95% CI) | NLL difference (95% CI) | Brier difference (95% CI) |
|---|---:|---:|---:|
| mean-MLP | -0.78 pp [-5.27, +3.91] | -.0190 [-.0338, -.0042] | -.0105 [-.0199, -.0016] |
| BiLSTM | +3.32 pp [-0.78, +7.42] | -.0173 [-.0433, +.0091] | -.0144 [-.0286, +.0004] |

Interpretation: the coverage increase improves static descriptive NLL/Brier
for both readouts and static Accuracy for BiLSTM in this single-seed contrast.
Temporal question-type intervals include zero for both architectures. Mean
confidence/ECE are recorded by task in each run report; calibration is mixed,
so the static NLL/Brier gains do not establish a uniformly calibrated model.

## Learning curve and overfitting

Training CE is the mean CE over that epoch's training set; it is not called a
fixed-set generalization gap. Validation values below are example-weighted
macro question-type Accuracy/NLL/Brier/ECE.

| Train scenes / readout | Step | Train CE | Val Accuracy | Val NLL | Val Brier | Val ECE |
|---|---:|---:|---:|---:|---:|---:|
| 256 / mean-MLP | 512 | .1789 | 40.82% | 2.0099 | .8802 | .3661 |
| 256 / mean-MLP | 1,024 | .0012 | 42.97% | 3.4592 | 1.0106 | .4664 |
| 256 / mean-MLP | 1,536 | .0003 | 43.95% | 3.8721 | 1.0232 | .4675 |
| 256 / mean-MLP | 1,920 | .0002 | 42.97% | 4.0541 | 1.0275 | .4750 |
| 256 / mean-MLP | 2,048 | .0002 | 43.36% | 4.1058 | 1.0297 | .4714 |
| 256 / BiLSTM | 512 | .0792 | 41.60% | 2.2462 | .9149 | .3934 |
| 256 / BiLSTM | 1,024 | .0010 | 43.36% | 3.3175 | .9880 | .4559 |
| 256 / BiLSTM | 1,536 | .0003 | 43.55% | 3.6641 | 1.0053 | .4666 |
| 256 / BiLSTM | 1,920 | .0002 | 43.75% | 3.8321 | 1.0109 | .4740 |
| 256 / BiLSTM | 2,048 | .0002 | 44.14% | 3.8842 | 1.0121 | .4713 |
| 512 / mean-MLP | 512 | .7682 | 47.07% | 1.2885 | .6766 | .1583 |
| 512 / mean-MLP | 1,024 | .1395 | 39.65% | 2.2578 | .9319 | .4163 |
| 512 / mean-MLP | 1,536 | .0090 | 39.84% | 3.2926 | 1.0164 | .4814 |
| 512 / mean-MLP | 1,920 | .0008 | 40.04% | 3.7851 | 1.0385 | .4845 |
| 512 / mean-MLP | 2,048 | .0006 | 40.04% | 3.8650 | 1.0432 | .4874 |
| 512 / BiLSTM | 512 | .5458 | 45.31% | 1.4191 | .7082 | .2182 |
| 512 / BiLSTM | 1,024 | .0202 | 41.99% | 2.8087 | .9461 | .4313 |
| 512 / BiLSTM | 1,536 | .0011 | 42.97% | 3.5222 | .9689 | .4532 |
| 512 / BiLSTM | 1,920 | .0006 | 43.16% | 3.7484 | .9797 | .4646 |
| 512 / BiLSTM | 2,048 | .0005 | 43.36% | 3.8126 | .9814 | .4646 |

Both datasets drive training CE near zero while validation NLL, Brier and ECE
degrade strongly. Validation Accuracy is comparatively flat after the first
512 updates. This is clear overfitting, not evidence that more updates improve
the final model. Best checkpoints are retained separately from final
checkpoints; no last-checkpoint selection was used.

## Execution record and deviations

- Both completed arms per dataset received 2,048 optimizer updates and
  131,072 example presentations. Unique examples/assets were 1,024/512 for
  the larger train pool and 512/256 for the control, causing 130,048 and
  130,560 repeated presentations after each arm's first unique pass. Validation
  was 512 questions /256 scenes in both. Validation selection means bootstrap
  intervals are descriptive conditional comparisons, not independent-test
  inference.
- 512-scene paired fit wall time: 73.56 seconds; readout training time
  mean-MLP 24.82 s / BiLSTM 31.44 s; eval plus serialization 7.32 s /7.17 s.
- 256-scene paired control wall time: 70.11 seconds. The CPU run reports per
  model training time mean-MLP 24.32 s / BiLSTM 31.57 s and eval plus
  serialization 5.99 s /5.96 s. GPU allocation was zero.
- Feature extraction was about 17.4 minutes elapsed. This run's complete local
  compute bill is $0; no cloud was used.
- First 512-scene feature extraction attempt failed on missing Flax. The
  isolated extractor environment was pinned to compatible Flax 0.11.2; a
  subsequent CPU-only extraction passed. The initial feature source code is
  the pinned VideoPrism checkout, not a modified repository copy.
- Initial preflight used an installed package instead of the worktree's `src`;
  this was corrected by setting `PYTHONPATH=src`. Preflight then exposed a
  variable-shadowing bug and fixed it.
- Initial fit attempt 01 stopped before the first optimizer update because
  CUDA memory reset was called for a CPU device. That attempt's config and
  input preflight are preserved at the `fit` subdirectory and are not used as
  results. A CPU-specific guard and regression test were committed; successful
  results are in fresh `fit-attempt-02` directories.
- Ruff and `tests/test_textfree_video.py` pass (12 tests after fixes). No
  repository-wide test run or CI result is claimed here. The branch is pushed;
  no PR was created and nothing was merged.
- No base model, encoder, prior cache, corpus or earlier checkpoint was
  overwritten. New report/config/code are on `codex/textfree-video-coverage-v1`.

## Decision and next step

**Coverage result: supports a static descriptive data-coverage benefit in this
readout setup; temporal coverage benefit remains uncertain.** The same-direction
static NLL/Brier changes for both architectures are consistent with the earlier
text-conditioned 256-vs-768 CLEVRER result, but the models and input contract
differ. No evidence here says that temporal reasoning is solved by adding
static/template examples. Recurrent readout shows no reliable temporal gain;
do not generalize that to all LSTM/xLSTM-style architectures.

Do not run another 2,048-update budget on these same scenes. A next video task
should add genuinely temporal supervision (explanatory/predictive/counterfactual)
while keeping scene-grouped validation and providing evidence that the sampled
frames contain the information needed for each label. The 95–98% target for
text/image/audio/video and the pre-quantization <=200M parameter goal remain
open; these small readouts are diagnostics, not a product candidate.

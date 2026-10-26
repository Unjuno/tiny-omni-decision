# DROID next-action direction: frozen video + sensor readouts (2026-10-26)

## Result and decision

This is a small, development-only robot action prediction task. It does not
establish broad robot competence, prove that both modalities are required, or
close ROADMAP R2 M-2. No sealed audit was loaded. The earlier DROID
source-folder success/failure experiment is unchanged and remains a negative
result.

The input is the same eight early VideoPrism frame features and eight aligned
robot-state samples used by the pinned DROID early-outcome task. The target is
the command at sample index `max(input_indices) + 1`, mapped to one of seven
choices: `stop`, `x-`, `x+`, `y-`, `y+`, `z-`, or `z+`. Direction is the signed
largest-magnitude component among the first three commanded Cartesian velocity
values; `stop` applies when all three absolute values are at most `1e-4`. The
official DROID schema describes `action_dict/cartesian_velocity` as commanded
Cartesian velocity. It supplies no per-step timestamps, so “next step” is an
index-based target, not a fixed-duration forecast. [Official DROID schema and
data access](https://github.com/droid-dataset/droid/blob/main/docs/the-droid-dataset.md)

The task has 337 train episodes and 200 development-validation episodes, with
zero recording-group overlap. Validation contains 18–80 examples per class.
The train-prior majority baseline is 40.0% validation Accuracy. The fixed-L2
attempt overfit. A follow-up selected regularization by five-fold
class-stratified group cross-validation on train only; it improved the readouts,
but the fused model still lost to the sensor-only model.

| Attempt / readout | Train Accuracy | Val Accuracy | Macro Val Accuracy | Val NLL | Macro Val NLL | Val Brier | Val ECE-10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train-prior baseline | — | 40.0% | 14.3% | 1.660 | 2.148 | 0.760 | 0.053 |
| 01 fixed L2, video | 100.0% | 32.0% | 19.6% | 3.752 | 5.176 | 1.118 | 0.475 |
| 01 fixed L2, sensor | 78.0% | 37.5% | 23.5% | 2.281 | 3.050 | 0.849 | 0.246 |
| 01 fixed L2, video + sensor | 100.0% | 33.0% | 19.5% | 3.613 | 5.091 | 1.111 | 0.487 |
| 02 train-CV L2, video | 95.5% | 33.5% | 16.9% | 1.935 | 2.648 | 0.844 | 0.216 |
| 02 train-CV L2, sensor | 50.7% | **44.0%** | **21.2%** | **1.593** | **2.037** | **0.741** | **0.069** |
| 02 train-CV L2, video + sensor | 97.9% | 36.5% | 19.8% | 1.901 | 2.626 | 0.834 | 0.208 |

The fixed train-CV grid was `{0.1, 1, 10, 100}` with seed 17. Macro NLL was
the selection metric and all three heads selected the upper boundary, `100`.
Validation was not used to select regularization. The train-CV result suggests
that the readouts remain data-limited/overfit; it does not justify further
architecture or gate work on this weak task.

In attempt 02, paired bootstrap over the 200 independent validation episodes
(5,000 resamples, seed 17) gave fusion-minus-sensor Accuracy delta `-7.5 pp`
with 95% percentile interval `[-14.0, -1.0] pp`; the NLL delta was `+0.310`
with interval `[+0.154, +0.466]`. Fusion-minus-video intervals included zero
for both Accuracy (`[-1.0, +7.0] pp`) and NLL (`[-0.094, +0.025]`). The
predeclared comparison therefore does not support useful video+sensor fusion.

Input controls show that the fused head is sensitive to paired observations,
but sensitivity alone is not evidence of correct fusion. With the trained
attempt-02 fused head, removing video changed validation Accuracy from 36.5%
to 40.5% (macro Accuracy fell from 19.8% to 15.9%); removing sensor changed
Accuracy to 35.5%. A fixed cross-episode derangement reduced Accuracy to
19.5% for video and 20.5% for sensor, but NLL rose above 16 because the head
became extremely overconfident. One-sampled-index stale/advanced sensor
controls scored 21.5%/22.0% Accuracy with NLL above 16. The different-target
class sensor donor control scored 21.5% Accuracy. These are stress controls on
the same exposed development set, not independent confirmation that both
streams are necessary; index shifts cannot be interpreted as seconds.

**Disposition:** retain the CPU readout implementation and all attempts for
reproducibility. Do not promote this DROID task or claim M-2. The sensor-only
readout is a weak task-specific reference, not a product model. No more
readout capacity, seed, or validation tuning is warranted from this result.

## Data, features, and integrity

- Source: DROID 1.4.0 official train shards, CC-BY-4.0 release record. The
  selected-shard manifest is SHA-256
  `2bfb5626d203344bf2bd976f711c0ed5f7b4c1979932b7b48565329099207e64`;
  download receipt is
  `2bfb6322e9c0c4649f875d8ad37fb3e990f786324fe2fdbd9cd6b2872af50667`.
- Group-safe task manifest SHA-256:
  `b882b66f8be2dd38aa7d69badd3d00b93f58769a9daa2be5f38088fbe084ec80`.
  Source records were matched by canonical episode path; duplicates, missing
  IDs, group overlap, shard counts, and the original feature-cache ID/split/
  label alignment fail closed.
- Frozen VideoPrism-B revision:
  `be719a406d563b66f0ac969e7c94bab8e997c81a`; checkpoint SHA-256
  `61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29`;
  114,365,184 parameters. Frozen cached features are `[537,3,8,768]` video
  and `[537,8,14]` sensor arrays. Their hashes are `d97a7f28bcdc119cccf4ef0d2019ae69852734b43a10d9102d320c0ff3043878` and
  `6ef3031c5521d1e40d8f19dd527e0709abeab0af44748f6a90416c15316d1016`.
- No media/feature cache, pretrained checkpoint, or sealed audit was changed.
  The early-outcome cache’s source label is used only to verify feature IDs and
  splits; it is not an input to the action readout.

## Model and resource accounting

Each arm is a deterministic multinomial L2 logistic readout. Video averages the
three views and concatenates the means of the first and last four sampled
frames; sensor flattens eight states × 14 channels; fusion concatenates both
summaries. Standardization is fit on the 337 training rows only. Attempt 01
uses fixed L2 strength 1. Attempt 02 uses the train-only group-stratified
cross-validation selection above, then refits once on all train rows. Every
validation prediction is saved; no checkpoint or model was selected by a
sealed test.

| Arm | Readout parameters | Total resident parameters including VideoPrism |
|---|---:|---:|
| Video | 10,759 | 114,375,943 |
| Sensor only | 791 | 791 |
| Video + sensor | 11,543 | 114,376,727 |

Attempt 02’s fused readout file is 114,290 bytes; the VideoPrism checkpoint is
457,475,296 bytes, so the minimal fused weight payload is 457,589,586 bytes
before runtime dependencies. This is above the provisional 200 MB package
target. No encoder training, GPU, cloud compute, or paid resource was used.
Attempt 01 completed in 24.8 seconds; attempt 02 completed in 63.1 seconds.
Both fits used Python 3.11.16, NumPy 1.26.4, and SciPy 1.17.1.

## Reproduction and artifacts

Code commits:

- `c33f6ab13d3fa55d87d039e88de63b564264106f`: fixed-L2 readout, integrity
  checks, and CPU tests.
- `faa841e0b17f6544ecc8aee7b6e3bbac85291258`: train-only group-CV selection.

The two immutable attempts are outside Git:

| Artifact | Attempt 01 fixed L2 | Attempt 02 train-only CV |
|---|---|---|
| Directory | `C:\CodexArtifacts\pretrained-reuse-r2\droid-next-action-readouts-20261026-attempt-01` | `C:\CodexArtifacts\pretrained-reuse-r2\droid-next-action-readouts-20261026-attempt-02` |
| Report SHA-256 | `bac6af6b4960ffe41b3137a773f53418353689698f5e791425a56546f32f4422` | `03aa177529db0e3d8a8c34ad2ac61c967cabca836e102366652159a7d1378af7` |
| Target manifest SHA-256 | `289bbbd9fed79199fb3835c5b903bf0a921c1e122c12083b95fc870b955b7d04` | same |
| Validation predictions SHA-256 | `46588f3e20d73557f406fbe52e57160c8aa9ef19c87e555cce84c0afb0e61065` | `43299c9557afb21778bf670651678091b0babc01445931325650712b6cdbf5ef` |
| Fused readout SHA-256 | `4ef300e1b0f74099efe5ee580225506f15f41cb0936779af74a7bc9198b7c342` | `7451e761b9be06bc9803f88941754b7e09d2aa3ccface909f59f0d0beef426bc` |

The two invocations used the same dataset/task/cache/checkpoint arguments and
distinct output directories. The runner recomputes target labels from the
hash-pinned TFRecords, verifies the original cache hashes and IDs, and refuses
to overwrite any output directory.

## Tests and remaining gates

All 46 DROID-named project tests pass, including the 15 focused action-readout
tests; full-project Ruff passes. Tests cover action label mapping, malformed
inputs, multimodal feature alignment, probability metrics, deterministic group
folds, and train-only CV selection. The report itself and per-example
prediction files are saved with the output artifacts above.

Open: a valid paired AV/sensor dataset and task, proof of both-input necessity
on a meaningful task, stronger independent task coverage, text/image/audio
integration, R2’s full M-0–M-6 battery, architecture comparison R3, diagnosed
alignment R4, full state-reuse R5, optional merge R6, measured low-bit package
and device runtime R7, and locked release-quality evaluation R8. This negative
readout result narrows one DROID task path; it does not establish a ceiling on
VideoPrism or multimodal decision models.

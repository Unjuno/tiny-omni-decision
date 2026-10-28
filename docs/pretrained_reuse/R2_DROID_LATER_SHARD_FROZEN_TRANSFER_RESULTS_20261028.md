# DROID later-shard frozen-readout transfer results

## Result and scope

Three existing DROID readouts were replayed unchanged on 137 eligible episodes
from previously unopened official-train shards 00466, 00759, and 01632. The
episode and recording-group IDs are disjoint from the original task, and the
task builder found zero sampled-frame content overlap. The source-only audit
had found no new collection sites; all 14 sites in this sample were already
represented in the prior cohort. This is a same-site, group-disjoint transfer
check, not a blind audit or new-site test.

No model was trained, selected, or calibrated. The video-only, sensor-only,
and video-plus-sensor L2 logistic readouts are the exact files from the prior
development run. All predictions use the same seven options and fixed
VideoPrism feature pipeline. The official-train sample was opened only after
the existing readout hashes and evaluation procedure had been frozen. No
sealed audit was loaded.

## Evaluation protocol

- Source: Google Research DROID 1.0.0, official train; object generations,
  byte sizes, license, and shard SHA-256 hashes are in the source manifest.
- Sample selection: salt `droid-r2-unseen-sites-v1`; lowest three SHA-256 ranks
  among shard indices 23–2047 with minimum spacing 20. Selected indices 466,
  759, and 1632. The source-only audit receipt is SHA-256
  `738231fb8bb100ae0d000a663ae6dee824d42bf3dd5dfe8e1c4e3f75ea1dad4d`.
- Task: use eight uniformly indexed observations from the first quarter of an
  episode, three camera views and 14 proprioception channels; predict the
  dominant signed Cartesian direction of the next commanded action, or stop.
  One short episode was excluded by the same minimum-length rule, leaving 137.
- The new sample has 56 `x+`, 37 `z-`, 14 `z+`, 12 `stop`, 9 `y-`, 6 `y+`, and
  3 `x-` targets. The largest-class baseline is therefore 40.88% Accuracy.
- Metrics use the fixed seven-class option order. ECE uses ten equal-width
  confidence bins. Results are descriptive on this development sample; no
  model decision was made from them.

## Overall metrics

| Readout | N | Accuracy | Macro Accuracy | NLL | Macro NLL | Brier | ECE-10 | Mean confidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Video | 137 | 39.42% | 18.58% | 1.8140 | 2.7859 | 0.8012 | 0.1546 | 0.5487 |
| Sensor | 137 | **42.34%** | **19.41%** | **1.4968** | **2.1788** | **0.7009** | **0.0429** | 0.4531 |
| Video + sensor | 137 | 39.42% | 18.45% | 1.7790 | 2.7407 | 0.7896 | 0.1954 | **0.5895** |

On these same 137 examples, adding video to the sensor readout decreased
Accuracy by 2.92 percentage points, increased NLL by 0.2822, increased Brier
by 0.0887, and increased ECE-10 by 0.1525. The higher joint confidence did
not correspond to better decisions. Video alone scored below the 40.88%
largest-class baseline; sensor-only was 1.46 points above it. Macro Accuracy
remained low for all arms.

| Shard | N | Video Accuracy | Sensor Accuracy | Joint Accuracy | Video NLL | Sensor NLL | Joint NLL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 466 | 49 | 32.65% | 38.78% | 36.73% | 2.0376 | 1.5457 | 1.9473 |
| 759 | 47 | 44.68% | 48.94% | 40.43% | 1.6262 | 1.4716 | 1.6797 |
| 1632 | 41 | 41.46% | 39.02% | 41.46% | 1.7622 | 1.4671 | 1.6916 |

Per-site Accuracy varies substantially and small sites have only two to four
examples. These site rows are descriptive, not stable site-level estimates.

| Site | N | Video | Sensor | Video + sensor |
| --- | ---: | ---: | ---: | ---: |
| autolab | 13 | 23.08% | 23.08% | 15.38% |
| bvl | 8 | 50.00% | 50.00% | 50.00% |
| clvr | 3 | 33.33% | 33.33% | 33.33% |
| guptalab | 2 | 0.00% | 0.00% | 0.00% |
| iliad | 14 | 28.57% | 28.57% | 21.43% |
| iprl | 11 | 27.27% | 36.36% | 36.36% |
| iris | 15 | 33.33% | 46.67% | 40.00% |
| pennpal | 3 | 33.33% | 33.33% | 66.67% |
| rad | 3 | 33.33% | 66.67% | 33.33% |
| rail | 13 | 61.54% | 61.54% | 69.23% |
| real | 3 | 66.67% | 100.00% | 66.67% |
| rpl | 4 | 50.00% | 50.00% | 25.00% |
| tri | 39 | 46.15% | 43.59% | 43.59% |
| weird | 6 | 33.33% | 33.33% | 33.33% |

Class-level performance is concentrated in `x+`; no arm correctly predicted
`stop`, `x-`, or `y+` on this sample. Full per-class NLL, Brier, ECE, and each
prediction are preserved in the external JSON report.

## Comparison with the original development validation

The prior site-held-out validation contained 181 different episodes and had
already been observed during model development. The table shows both sets for
context only. They are **not paired** and differ in shard, site, and label mix;
the deltas below do not establish a quality increase or decrease.

| Readout | Set | N | Accuracy | Macro Accuracy | NLL | Macro NLL | Brier | ECE-10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Video | Prior validation | 181 | 35.91% | 18.61% | 1.9406 | 2.5715 | 0.8626 | 0.2207 |
| Video | Later-shard sample | 137 | 39.42% | 18.58% | 1.8140 | 2.7859 | 0.8012 | 0.1546 |
| Sensor | Prior validation | 181 | 37.02% | 17.41% | 1.5128 | 1.9712 | 0.7219 | 0.0904 |
| Sensor | Later-shard sample | 137 | 42.34% | 19.41% | 1.4968 | 2.1788 | 0.7009 | 0.0429 |
| Video + sensor | Prior validation | 181 | 34.81% | 18.49% | 1.9034 | 2.4589 | 0.8508 | 0.2447 |
| Video + sensor | Later-shard sample | 137 | 39.42% | 18.45% | 1.7790 | 2.7407 | 0.7896 | 0.1954 |

The fresh sample's `x+` frequency is 40.88%, versus 37.02% on the prior
validation, and its class/site composition differs. The modest raw Accuracy
changes across sets must not be read as evidence that the models improved.
The within-sample matched-arm comparison is more relevant here: sensor-only
remained better than video-plus-sensor on Accuracy and all reported proper
scoring/calibration metrics.

## Frozen artifacts and hashes

- Source manifest:
  `manifests/pretrained_reuse/droid-later-shard-frozen-transfer-source-20261028.json`,
  SHA-256 `8d664bfa59eb2c391d5a86eb47547a89d4569fd747e2f10b6e0127350a2dfd80`.
- Effective feature-extraction config (external): SHA-256
  `2f5ed3e8d18485ecb3de86f74c00bf6527d482077d8e8ff8ebf5b650ca6b6cb6`.
- Evaluation task manifest (external): SHA-256
  `3bc3eb7d5bba50a69d60e3eb020344b25d891be42bbcb19ec83ece12186ec57e`.
- Feature-cache summary (external):
  `C:\CodexArtifacts\pretrained-reuse-r2\droid-later-shard-transfer-features-attempt-02\summary.json`.
  Video features SHA-256 `41023816f9d70d6ce2cf0a0f621d311a389b0653813d40753ac6a1a7b6d3ec93`;
  sensor features `35c7c52325b8e0f57023d514707e10f93923259cc161e4be4ad3014180c08654`;
  labels `696dd2ea1ecbfa1302ced61292626c14e2ab040acad1782ab7e559ffc7d9f2ad`.
- Fitted readout hashes are unchanged from the prior report:
  video `4afefda7c2ab1d034f2d78f124aff393932071f11a319cc6274be20296f798fe`;
  sensor `056ced9e4d9653c231d51ba0290bd87b0983a14da34b70e0de00ab5ae7885b74`;
  video + sensor `dde026c9a8bb377d04ecb059971a3eb185891739ba367f001018575225f83a75`.
- Final report and predictions are external at
  `C:\CodexArtifacts\pretrained-reuse-r2\droid-later-shard-transfer-evaluation-attempt-01\`.
  Evaluation report SHA-256 `429baa6dcb41b713931e4f634a8ddcc671381424916fdbf6350e0ec84f3b9fff`.
- The 114,365,184-parameter / 457,475,296-byte VideoPrism checkpoint remains
  outside Git and unchanged.

## Runtime and deviations

Task building took 20.6 seconds. CPU VideoPrism extraction took 472.07 seconds
for 137 episodes (17.42 episodes/minute), close to the prior 17.66/minute
measured on 906 examples. It used JAX 0.6.2 and TensorFlow 2.18.1, with device
`TFRT_CPU_0`; no GPU, cloud, or paid compute was used. The final readout replay
took under one second.

The first extraction attempt used the sample directory directly, while the
existing extractor expects TFRecord basenames from the pinned object manifest.
It loaded VideoPrism on CPU, then stopped at the first missing filename before
decoding an episode; `progress.jsonl` remained empty. Its zero-filled feature
arrays are retained in the separate attempt-01 directory. Attempt-02 used
same-volume hard links with the expected names, so it did not duplicate the
3.17 GB source payload or modify any shard.

Code was committed before opening labels at `b8676aec` and the append-order
progress validation correction was committed at `7f0decb7`. The latter was
needed because the extractor appends progress in shard traversal order while
writing feature rows by stable task index. The evaluator now validates the
stable index-to-ID mapping instead of incorrectly assuming append order is
task-row order.

## Decision

This fresh episode-group evaluation corroborates the earlier result: the
sensor-only baseline remains stronger than the matched VideoPrism+sensor
readout on this DROID action-direction task. The video input is not shown to
improve the task's decision quality, despite the earlier within-site video
permutation sensitivity. This is a narrow result; it does not establish that
video is generally unhelpful or that temporal representations cannot help.
Do not change the readouts, promote the checkpoint, or claim product quality
from this comparison. This result alone does not justify fitting a temporal
model on the same observed development data. If a temporal model is tested
later, freeze its protocol first and reserve a different untouched evaluation
source or task.

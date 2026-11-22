# HABIT data scale-up result — 2026-11-22

## Decision

The fixed seed-17 readout comparison completed all 80 epochs in each of four
arms. The best checkpoint was epoch 2 in every arm. Expanding the same frozen
train/validation selection from 202/76 to 606/228 transitions did not produce
an observed quality gain on the 76 validation IDs shared with the earlier
pilot. Accuracy was unchanged or lower, while NLL and Brier were slightly
worse. This is weak development evidence: the expanded run selected its
checkpoint using all 228 validation rows, including the matched 76.

The scale-up helps test training-data volume, but it does not increase
independent validation task groups: both runs use the same six groups. It also
does not address independent people, institutions, audio, image, or a new
decision domain. Do not promote this readout or infer a model-family ceiling.

## Scope and fixed conditions

- Source revision: 3127c0d216d9304bc46ad17f00f422bb91f867d1.
- New experiment: habit-temporal-decision-cpu-seed17-data-scale-v1.
- Dataset: HABIT, revision bae97d251907743f8fb6db7e70c61d361df4dc1e,
  CC BY 4.0.
- Only the already-frozen train and development-validation episode selection
  was scaled. All original 66 train / 24 validation episode IDs were retained.
  No sealed-audit rows or payload were opened.
- The prior 80-epoch readout architecture, seed 17, batch size 32, learning
  rate 0.001, weight decay 0.0001, cross-entropy loss, randomized option
  order, train-only sensor normalization, four input modes, minimum validation
  macro task-group NLL selector, and no-early-stopping rule were kept.
- The intended change was data coverage and new output/config paths. The
  validation set grew within the same six task groups; therefore it is not an
  independent-group evaluation.
- No quantization, sealed-audit evaluation, or product promotion occurred.

## Integrity and artifacts

The expansion selection receipt SHA-256 is
2a644aff5c310c1607ac3c4f858351bf932b128af9eec79ec487cbaf5e3f46a9.
The frozen publisher inventory SHA-256 is
78695828206d318f26e84c19286cf47daceb4bddc420f613e6d745488a5e7ffe.
The payload audit SHA-256 is
c0f4eb251efbda4419b8fc438a6786092e9895964ab90c3fad67899bd1ecbe38.
It confirms 270 train/validation episode metadata rows, no cross-split video
hash overlap, no action-column reads, and no sealed-audit access.

The 540 media/Parquet payload files totaled 1,895,998,397 bytes and passed
publisher SHA-256 verification. The feature extraction used the pinned
VideoPrism-B checkpoint (457,475,296 bytes; SHA-256
61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29) and
upstream source commit c6eba0ece3dd49a24ffc44582966997a4a59f1d8.
The worker ran on CPU and produced 834 records: 606 train and 228 validation.
There are 198 train videos / 37 represented task groups and 72 validation
videos / 6 represented task groups.

An initial nested worker invocation was interrupted after its launcher process
was mistaken for an idle worker; the already-written input manifest was kept.
The worker was then run directly against that same manifest, pinned checkpoint,
and source revision and completed successfully. No training or source data
changed between these invocations.

Key immutable output paths are under
C:\CodexLocalArtifacts\pretrained-reuse-r2\habit-data-scaleup-seed17-attempt-01:

- selection.json
- inventory.json
- payload-audit.json
- features/result.json
- features/transition-inputs.json
- features/habit-transition-features.npz
- decision-package/package-report.json
- training/effective_config.yaml
- training/experiment-report.json
- training/predictions.json
- Four per-arm history, best-checkpoint, and last-checkpoint files
- training/matched-validation-comparison.json

Committed configuration:
configs/pretrained_reuse/habit_temporal_decision_data_scale.yaml.
Its SHA-256 is
8bf3056cb4752da15fc5fbec9c90c9394298bc7cdffe59d209d7a5b27991fde9.
Package-report SHA-256:
062013df3edf0c63c84e4c43edeb454430cf896a0ed75d5500011d2f492d505e.
Feature cache SHA-256:
5c494417aa0946041881f5a9305eb852d617769dc3261dda99c9772bb6301928.
Transition-example JSON SHA-256:
87b405ba343f4a251bd3450f306514e552a66b860b9cc558fd65880c09d2f8f5.

All model weights and media remain outside Git. The four trained readout arms
contain 196,865 trainable parameters for text-only and 370,819 for each arm
using the video/state readout. The recorded total unique pre-quantization
parameter count is 131,951,873 for text-only and 132,125,827 for the other
three arms. No claim is made about a quantized package size.

## Full expanded validation results

All rows below are development validation and were involved in checkpoint
selection. Accuracy is sample-weighted; macro-group values average the six
task groups equally.

| Arm | Best epoch | Accuracy | NLL | Brier | ECE (15 bins) | Macro-group accuracy | Macro-group NLL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Text/options | 2 | 0.4737 | 0.9973 | 0.5972 | 0.0563 | 0.4286 | 1.0045 |
| Video + text | 2 | 0.4737 | 0.9971 | 0.5973 | 0.0690 | 0.4286 | 1.0021 |
| State + text | 2 | 0.4737 | 0.9985 | 0.5981 | 0.0736 | 0.4286 | 1.0035 |
| Video + state + text | 2 | 0.4737 | 0.9955 | 0.5962 | 0.0643 | 0.4286 | 1.0006 |

Every arm completed all 80 epochs. From epoch 2 to epoch 80, training NLL
decreased in all arms while validation macro-group NLL increased:

| Arm | Train NLL, epoch 2 → 80 | Validation macro-group NLL, epoch 2 → 80 |
| --- | ---: | ---: |
| Text/options | 0.9246 → 0.8095 | 1.0045 → 1.6675 |
| Video + text | 0.9317 → 0.0000 | 1.0021 → 10.1591 |
| State + text | 0.9294 → 0.2830 | 1.0035 → 1.5387 |
| Video + state + text | 0.9309 → 0.0000 | 1.0006 → 8.3142 |

This is clear overfitting on the exposed development validation. The best
epoch is an early selector result, not evidence that more epochs help.

## Matched comparison on the original 76 validation IDs

All original IDs were found in the expanded validation set. For every arm,
sample IDs, task-group IDs, targets, and option lists matched exactly. The
following compares the old pilot's selected checkpoint with the expanded
run's selected checkpoint evaluated on those same 76 rows.

| Arm | Accuracy old → scaled | NLL old → scaled | Brier old → scaled | ECE old → scaled |
| --- | ---: | ---: | ---: | ---: |
| Text/options | 0.5263 → 0.4737 | 0.9931 → 0.9973 | 0.5946 → 0.5972 | 0.0580 → 0.0563 |
| Video + text | 0.5263 → 0.4737 | 0.9936 → 0.9972 | 0.5956 → 0.5974 | 0.0677 → 0.0690 |
| State + text | 0.4868 → 0.4737 | 0.9941 → 0.9987 | 0.5960 → 0.5983 | 0.0786 → 0.0739 |
| Video + state + text | 0.4737 → 0.4737 | 0.9907 → 0.9955 | 0.5936 → 0.5963 | 0.0833 → 0.0643 |

For each mode, the six-group paired bootstrap used seed 17 and 2,000
resamples. The macro-group accuracy difference (scaled minus old) was
−0.0833 for text/options (95% percentile interval [−0.25, 0.00]),
−0.0833 for video + text ([−0.25, 0.00]), −0.0208 for state + text
([−0.0625, 0.00]), and 0.0000 for video + state + text ([0.00, 0.00]).
The macro-group NLL differences were respectively +0.0086
([−0.0040, +0.0292]), +0.0086 ([−0.0015, +0.0199]), +0.0099
([−0.0017, +0.0229]), and +0.0085 ([+0.0013, +0.0162]).

The bootstrap has only six groups, and both checkpoints were selected on
development validation. The intervals are diagnostic, not a blind-test
confidence statement. The matched comparison does not support a quality gain
from this data scale-up.

## Runtime and interpretation

Feature extraction reported 1,409 seconds on CPU (202 seconds video decode,
1,189 seconds model forward) and about 1.93 GB working set at exit. Training
reported 324.1 seconds, including 6.2 seconds text feature extraction, with
four CPU threads and about 762 MB working set at exit. PyTorch 2.6.0+cu124
was installed, but the run used CPU and CUDA was not initialized. No cloud
compute or paid compute was used.

The experiment is limited to text, video, and robot-state decision inputs.
HABIT has no audio input in this setup. Expanded validation rows remain within
six task groups, and the source comes from one institution and roughly ten
collectors. The evidence therefore does not establish robustness to new
people, institutions, tasks, or modalities, and it does not answer the
project's four-modality quality target.

## Next step

Do not spend more epochs or seeds on this fixed pilot. Before another fit,
admit additional rights-cleared data with genuinely new held-out task groups
and a clear decision label, or pivot to a better-grouped decision source.
Keep the current results as negative/limited data-scale evidence. Keep the
four-modality and pre-quantization parameter-count goals unchanged.
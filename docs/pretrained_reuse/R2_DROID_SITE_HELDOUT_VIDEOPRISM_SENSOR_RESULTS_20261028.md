# R2 DROID site-held-out VideoPrism and sensor readout

## Outcome

This is a development-only, one-step robot Cartesian action-direction probe.
It does not demonstrate general Omni decision quality or meet the R0 quality
target. A frozen sensor-only linear readout was stronger than video-only and
video-plus-sensor on held-out sites. Video features contain some aligned
signal, but adding them did not improve the main metrics over sensor-only.
This is not a product candidate, blind audit, or representation ceiling.

## Frozen source and task

- Source: Google Research DROID release 1.0.0, official train partition,
  shards 00000 through 00019; declared license CC-BY-4.0.
- Source manifest SHA-256:
  6ac6b222b43b1ff889952d880beaab6fbf0d3129c493d2856f98f8289d9588ec
- Prior train/validation manifest SHA-256:
  b7315c23bad4f9b195828c934cc3a87aeb34d36b1b1180a8e8152f783148e21d
- Frozen site split protocol SHA-256:
  2b88549b6fe6dd344841b3fd2ca55dfbf49e4cc923f12f70559d54a59338999c
- Frozen task manifest SHA-256:
  6f32c4d4a2ad1e2fed6ac1c904010c158cd6f17366b5f049c59dba27e6fef6cd
- Source records: 911; eligible examples: 906; five short episodes excluded
  under the fixed minimum-observation rule.
- Train: 725 examples from ten collection sites.
- Development validation: 181 examples from four disjoint sites
  (bvl, iliad, pennpal, real). Site choice used eligible site counts, not labels.
- Source episode/recording-group overlap: zero. Exact sampled JPEG hash
  overlap across splits: zero. Unique sampled frame hashes: 17,400 train and
  4,344 validation.
- The source-only cohort feasibility scan had inspected labels for 149 records,
  including 34 records from the eventual validation sites. Validation is
  therefore nonblind development data even though site assignment did not use
  labels. No sealed audit data were loaded.

The target is one of seven fixed labels: stop, x-, x+, y-, y+, z-, or z+,
derived from the dominant signed xyz component of commanded Cartesian velocity
at the first source step after the observed window. Inputs are eight uniformly
indexed observations from the first quarter of an episode: three camera views
and 14 proprioceptive channels. No future observation is an input; no
wall-clock temporal spacing is asserted.

## Representation and fitting

- Video encoder: google/videoprism-base-f16r288, revision
  be719a406d563b66f0ac969e7c94bab8e997c81a.
- Checkpoint: 114,365,184 parameters / 457,475,296 bytes;
  SHA-256 61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29.
- Feature cache: video [906, 3, 8, 768]; sensor [906, 8, 14].
- Video readout input: average the three view features, then concatenate means
  of frames 0–3 and 4–7. Sensor input: flatten the fixed 8-by-14 sequence.
  The joint arm concatenates these representations.
- Encoders were frozen; only three L2 multinomial logistic readouts were fit.
  Standardization uses train only. L2 strength was selected from
  {0.1, 1, 10, 100} by five-fold collection-group stratified CV on train sites,
  seed 17, macro NLL. All three arms selected the upper boundary, 100; no
  validation result was used for selection.
- The original 500-iteration solver cap did not converge in training-only CV.
  A separate readout config used a 2,000-iteration cap; the original config is
  unchanged. This is a solver convergence adjustment, not a change to data,
  features, or validation selection.
- Local runtime: CPU-only Intel Core i7-12700H, 14 cores / 20 logical
  processors. Pinned VideoPrism environment: JAX 0.6.2, TensorFlow CPU 2.18.1;
  readout used existing SciPy 1.17.1. No packages were upgraded. The active
  Orbitune process retained the GPU; this work did not use or stop it.
- Three full-shard transfer preflights averaged 25.3 MB/s. That rate was not a
  useful predictor for feature extraction: the actual 906-row extraction
  averaged 17.66 examples/minute over the complete run. During execution,
  three initial 60-second progress samples averaged 23.33 examples/minute;
  later intervals varied, so the wait estimate was recalculated from observed
  progress instead of relying on the initial projection.
- Video feature extraction completed 906 examples in 3,078.37 seconds
  (51.31 minutes). Source/task rebuild audit took 158.83 seconds. Final
  readout fit and evaluation took 80.77 seconds. No cloud or paid compute.

## Results

All figures use the same 181 nonblind development validation IDs and fixed
seven-choice target. ECE-10 uses ten confidence bins.

| Readout | Accuracy | Macro Accuracy | NLL | Macro NLL | Brier | ECE-10 | Mean confidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Video | 0.3591 | 0.1861 | 1.9406 | 2.5715 | 0.8626 | 0.2207 | 0.5658 |
| Sensor | **0.3702** | 0.1741 | **1.5128** | **1.9712** | **0.7219** | **0.0904** | 0.4549 |
| Video + sensor | 0.3481 | **0.1849** | 1.9034 | 2.4589 | 0.8508 | 0.2447 | 0.5928 |
| Train class prior | 0.3702 | 0.1429 | 1.6623 | 2.2371 | 0.7688 | 0.0491 | 0.4193 |

Validation accuracy by site:

| Site | N | Video | Sensor | Video + sensor |
| --- | ---: | ---: | ---: | ---: |
| bvl | 31 | 0.452 | 0.419 | 0.419 |
| iliad | 67 | 0.224 | 0.254 | 0.224 |
| pennpal | 32 | 0.312 | 0.344 | 0.281 |
| real | 51 | 0.510 | 0.510 | 0.510 |

Equal-weight means across the four sites:

| Readout | Accuracy | Macro Accuracy | NLL | Macro NLL | Brier | ECE-10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Video | 0.3744 | **0.2075** | 1.9717 | 2.4678 | 0.8594 | 0.2751 |
| Sensor | **0.3817** | 0.1900 | **1.4983** | **1.9352** | **0.7147** | **0.1523** |
| Video + sensor | 0.3586 | 0.1838 | 1.9222 | 2.3709 | 0.8485 | 0.2709 |

Paired site bootstrap: 5,000 resamples, seed 17, four validation sites.
For joint minus video, the Accuracy interval was [-0.0317, 0.0000] and NLL
[-0.0866, -0.0106]. For joint minus sensor, Accuracy was
[-0.0472, 0.0000] and NLL [0.2038, 0.6490]. Four sites provide weak
uncertainty resolution; these are diagnostic intervals, not population-level
claims.

For an input-use control, validation video features were permuted between
examples within each site while sensor features and the fitted joint model
were held fixed. Joint Accuracy fell from 0.3481 to 0.3149; NLL rose from
1.9034 to 2.0860. This indicates some aligned video contribution, but does
not show a useful complementarity benefit over sensor-only.

Validation labels are imbalanced (x+: 67/181; z-: 46/181; stop: 28/181;
other classes: 7–14 each). Pooled accuracy equals the train-prior baseline,
and macro metrics remain weak. Task difficulty, label semantics, limited
feature summary, sample count, and cross-site shift are plausible contributors;
this probe does not isolate one cause.

## Execution record and artifacts

Source code was pinned from commit
eba818067c6a9ef9089288205c26f6a8254ca6e5; work is isolated on branch
codex/r2-droid-expanded-sites-20261028.

The first task rebuild exposed a stale loop variable: each shard's record count
was compared with the previous loop's last shard. The check now uses the current
shard. A byte-hash mismatch was then traced to Windows CRLF versus LF
serialization. The builder writes CRLF, and the final source rebuild
reproduced the frozen task hash exactly while passing ID/group/frame overlap
gates.

Feature arrays and summary are outside Git at
C:\CodexArtifacts\pretrained-reuse-r2\droid-expanded-site-holdout-features-attempt-01.
Status is PASS, all 906 examples are present, backend is TFRT_CPU_0, and both
sealed_audit_loaded and model_training are false. Feature hashes:

- Video: 3d0308664e48c2613abef6116d7b61d2c3d7996b7efb20cd17e57c88a0465d0f
- Sensor: bbe1da48f6103bd0c3ce1e9cb9f86a3a3d28dd04fbde40e942f2c44664414e70
- Labels: 2ceabd1e99552c8b6864cc6d80015088d4a68d1a2f5079710e3d0d702e9ce841

Evaluation report, validation predictions, and fitted readouts are outside Git
at C:\CodexArtifacts\pretrained-reuse-r2\droid-expanded-site-holdout-evaluation-attempt-03.
Report status is PASS_EVALUATED. Validation prediction SHA-256:
2068212ffd29a99b719d81cc54b3686656cee53deab1f3d1bc3b12b2ecf850a3.
No feature arrays, source media, or model weights are in Git.

## Decision

Keep VideoPrism as a frozen representation reference, not an adopted product
component from this probe. Sensor-only performed best on NLL, Brier, and
calibration; the joint readout was weaker than sensor-only. Do not promote
this result into the four-modality quality claim. A follow-up needs a
prospectively defined task where visual state is essential, more independent
collection sites, and temporal features beyond two coarse pooled windows. Do
not tune the observed validation set.

## Reproduction commands

The final source/task audit used a new external output directory:

~~~powershell
py -3.11 -X utf8 scripts\build_droid_expanded_site_task.py --dataset-root C:\CodexArtifacts\pretrained-reuse-r2\droid-expanded-site-holdout-data-attempt-01 --selection-manifest manifests\pretrained_reuse\droid-expanded-site-holdout-source-20261028.json --previous-task-manifest C:\CodexArtifacts\pretrained-reuse-r2\droid-fresh-next-action-prep-attempt-02\task-manifest.json --split-protocol manifests\pretrained_reuse\droid-expanded-site-holdout-split-protocol-20261028.json --output-dir C:\CodexArtifacts\pretrained-reuse-r2\droid-expanded-site-holdout-rebuild-attempt-04
~~~

The frozen feature extraction used the pinned VideoPrism environment and wrote
only to the external feature-cache directory:

~~~powershell
py -3.11 -X utf8 scripts\extract_droid_fresh_video_sensor_features.py --dataset-root C:\CodexArtifacts\pretrained-reuse-r2\droid-expanded-site-holdout-data-attempt-01 --manifest manifests\pretrained_reuse\droid-expanded-site-holdout-source-20261028.json --task-manifest manifests\pretrained_reuse\droid-expanded-site-holdout-task-20261028.json --config configs\pretrained_reuse\droid_expanded_site_holdout_v1.json --checkpoint C:\CodexArtifacts\pretrained-reuse-r1\videoprism-base-f16r288.npz --model-source C:\CodexArtifacts\pretrained-reuse-r1\videoprism-base-r1\source --output-dir C:\CodexArtifacts\pretrained-reuse-r2\droid-expanded-site-holdout-features-attempt-01
~~~

The successful readout evaluation used the existing VideoPrism environment
because the global Python lacked SciPy; no environment was changed:

~~~powershell
& C:\CodexArtifacts\pretrained-reuse-r1\videoprism-base-r1\venv\Scripts\python.exe -X utf8 scripts\evaluate_droid_fresh_next_action.py --task-manifest manifests\pretrained_reuse\droid-expanded-site-holdout-task-20261028.json --feature-cache C:\CodexArtifacts\pretrained-reuse-r2\droid-expanded-site-holdout-features-attempt-01 --checkpoint C:\CodexArtifacts\pretrained-reuse-r1\videoprism-base-f16r288.npz --config configs\pretrained_reuse\droid_expanded_site_holdout_v1.json --readout-config configs\pretrained_reuse\droid_expanded_site_holdout_readout_v1_solver2k.json --output-dir C:\CodexArtifacts\pretrained-reuse-r2\droid-expanded-site-holdout-evaluation-attempt-03
~~~

The full CPU test suite completed with 2,870 passed and 11 skipped; Ruff passed
for all new or copied Python files.

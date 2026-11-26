# Gated option readout: fixed real-data comparison

Protocol/runner frozen at **5bffb02** before execution. A single seed17 CPU run
completed100epochs on the unchanged130training/39validation VIDIMU trials.
Best is epoch100 by the fixed validation NLL selector; exact saved-prediction
reload passed. Old caches, weights, split and results were read-only.

| Selected readout | Accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|
| Sensor linear | 1.000000 | 0.163704 | 0.065464 | 0.135254 |
| Video linear | 0.666667 | 0.776570 | 0.436735 | 0.131427 |
| Block-scaled linear fusion | 0.871795 | 0.350965 | 0.199967 | 0.092742 |
| New gated option scorer | 0.897436 | 1.729142 | 0.732459 | 0.706161 |

**Do not adopt.** A one-example Accuracy improvement over block-scaled fusion
coexists with much worse probability quality. Sensor-only remains better. The
high ECE is underconfidence here, not proof of overconfident wrong decisions.
Mean maximum predicted probability is0.191275 against Accuracy0.897436.

| Epoch | Train NLL | Validation NLL | Validation Accuracy |
|---|---:|---:|---:|
| 0 | 2.565056 | 2.564976 | 0.076923 |
| 25 | 2.433904 | 2.443633 | 0.589744 |
| 50 | 2.238526 | 2.270496 | 0.769231 |
| 75 | 1.951809 | 2.024874 | 0.769231 |
| 100 | 1.606334 | 1.729142 | 0.897436 |

Both losses are still improving, with endpoint gap0.122807. This fixed-budget
run does not establish a plateau or a capacity ceiling. Do not explain away
the observed poor probability quality or silently extend its preregistered
budget. No post-hoc calibration, extra seeds or epoch extensions were executed.

## Same selected weights, fixed controls

| Condition | Accuracy | NLL |
|---|---:|---:|
| Matched | 0.897436 | 1.729142 |
| Shuffle video within placement | 0.307692 | 2.327246 |
| Explicit video absence | 0.743590 | 1.975414 |
| Shuffle sensor within placement | 0.538462 | 1.984412 |
| Explicit sensor absence | 0.717949 | 1.840117 |

Both inputs influence the selected model. Missingness is outside training;
explicit masks are not equivalent to the old linear zero-vector controls.
Subject-cluster paired NLL differences (new minus comparator),10,000 draws,
seed17, percentile95/linear:

- Sensor: +1.565438, interval[1.510873,1.611567].
- Video: +0.952572, interval[0.694137,1.186895].
- Scaled linear fusion: +1.378176, interval[1.251778,1.463972].

Only three development subjects; repeated validation and checkpoint selection
are not corrected by these intervals. No independent/sealed-audit claim.

## Numerical checks and failed checks retained

Controls attempts01 and02 stopped on exact candidate-permutation equality:
maximum probability difference5.9604645e-08, logit difference4.7683716e-07.
The model has no option-position features, but float32 batched kernels and
softmax reductions need not preserve bitwise equality under permutation.
Attempt03 records both differences, checks logits at rtol/atol1e-6, probabilities
at rtol1e-6/atol1e-7, and verifies all39 answer identities agree. Tolerances were
introduced after observing the failed exact checks and are disclosed here;
the earlier exact small-fixture claim is not generalized to real data.
Checkpoint reload under unchanged option order remains bitwise exact. No
weights were retrained to pass these checks. Failed output directories remain.

## Accounting and artifacts

Trainable parameters421,826; with frozen VideoPrism,114,787,010 total for this
video/sensor probe. This excludes other Omni towers and is not a complete
all-modality deployment count. Training/evaluation/save loop2.616seconds,
Windows process peak working set565,366,784bytes; cached encoders were not loaded
in this process. Local CPU only, no GPU/cloud charge. Training history records
finite gradient norms at every update. All101 prediction snapshots preserved.

Executed command from repository root:

```python
import runpy, psutil, json
from pathlib import Path
runpy.run_module('scripts.train_vidimu_fusion_arm', run_name='__main__')
info = psutil.Process().memory_info()
Path('C:/CodexArtifacts/pretrained-reuse-r2/vidimu-gated-fusion-arm-01/process-memory.json').write_text(json.dumps(info._asdict(), indent=2))
```

Comparison command: `python -m scripts.compare_vidimu_fusion`.

Artifacts under `C:/CodexArtifacts/pretrained-reuse-r2/`:

- `vidimu-gated-fusion-arm-01`: configuration/source hashes, normalization,
  records, learning history, predictions, best/final weights and measured memory.
- `vidimu-gated-fusion-controls-03`: all control probabilities/metrics and paired
  intervals; attempts01/02 preserved separately.

Best SHA256:`d40ebf85278f5b5d3277f809c7478d4edc116832de97f1459faeacd9da365fd7`.
Final SHA256:`262a4feba390ba8742d33f0c2f21bde368e13222027fa0e35653cbd0e6ac8fd3`.
Different serialization filenames produce different file hashes despite the
same selected/final epoch; selection/reload was verified from tensor outputs.
Every best/final state tensor was also checked for exact equality and passed.

This is actual paired-feature training and option scoring, but remains an
episode-level13-class task with a placement confound. It neither resolves
VIDIMU synchronization nor establishes a task that requires both streams.
Temporal evidence and genuinely complementary paired tasks remain the next
missing R2-M evidence; do not replace them with this Accuracy figure.

# PhysicalAI stack-platforms fixed Path A pilot results — 2026-11-02

## Disposition

Execution completed, but the source/task result is **UNCERTAIN** and remains
`REVIEW_NOT_ADMITTED`. The joint video+proprio readout scored 72.5% validation
Accuracy versus 70.4% for the best single input (proprio history), but the
paired episode-cluster 95% interval for the gain includes zero. The missing and
episode-misaligned controls show that the fitted joint readout responds to both
inputs, but they do not satisfy the predeclared requirement that joint quality
reliably exceed the best single-input result. The fitted heads also reached
100% training Accuracy, a large train/validation gap.

This is a narrow action-prototype prediction task, not a successful or optimal
robot policy and not evidence of a general Omni Decision model. No candidate is
promoted. The validation result is now observed; do not tune on these IDs.

## Frozen task and execution

The task and split follow the pre-existing [pilot protocol](R2_PHYSICALAI_STACK_PLATFORMS_PILOT_PROTOCOL_20261102.md): for each episode, use the last
eight synchronized world-camera frames and eight aligned 81-D state rows to
predict one of eight train-only action prototypes for the demonstrator action
three frames (100 ms at 30 fps) after the observation window. Four windows are
used per episode. The label is a quantized recorded action, not an optimality,
task-success, or deployment label.

- Source: NVIDIA `PhysicalAI-Robotics-Manipulation-SingleArm`,
  `panda-stack-platforms`, revision `73c00c18e7ca108e91846294b20a1d7cd5de9340`;
  publisher license is CC BY 4.0.
- Split: manifest v3, SHA-256
  `3a2849afaa3cefa414c9dbfc094976e0f1cc48ad0e9636d65573fa306d1ed967`;
  700 train and 250 validation episodes, 2,800 and 1,000 windows respectively.
  Episode IDs are disjoint. No reserved episode was downloaded or read.
- Before and after extraction, all 1,900 selected input files were checked
  against the acquisition receipt: 192,700,909 bytes, no missing files, size
  mismatches, or SHA-256 mismatches. Receipt SHA-256:
  `1a2b09ad683fe5da38fec74ac937aefaa9c5a1b2ed969285478a0f724f13a498`.
- The earlier [normalized-frame audit](R2_PHYSICALAI_STACK_PLATFORMS_FRAME_CONTENT_AUDIT_20261102.md)
  found 78,248 unique normalized frame hashes and zero exact cross-split
  matches. That does not rule out near-duplicates. Metadata does not provide a
  scene/layout/environment ID; a label-blind neighbor review confirms that
  train and validation include the same camera, robot, work surface, and scene
  geometry. Results are episode-held, same-scene development evidence.
- Video features use frozen FP32 DINOv2-base, revision
  `f9e44c814b77203eaa57a6bdbbd535f21ede1415`, mean-pooled across each eight
  frame window, with last-frame as a predeclared control. Mean pooling discards frame order, so this experiment does not test temporal modeling. Sensor input is the
  flattened eight-row state history. The readout is linear eight-class softmax
  with train-only normalization and full-batch LBFGS (`max_iter=200`), seed 17.
  Action normalization and the eight-prototype codebook use train data only.
  Validation was not used for checkpoint selection, calibration fitting, or
  hyperparameter search.
- Controls: proprio-only, video mean, video last-frame, joint, episode-shuffled
  video, missing video, and missing proprio. Missing inputs use the joint
  model's train feature mean; controls are not refit.
- The only model fit was the predeclared one-shot readout. The pretrained
  encoder was frozen. Its exact checkpoint has 86,580,480 parameters and a
  346,345,912-byte weight file; the fitted joint Linear head has 11,336
  parameters. The probe checkpoint contains all four fitted arms and is not a
  packaged or quantized product model.
- Runner code commit was `95b6514a7a23fa6e36985e766056aa332f044cd4` on
  `docs/pretrained-reuse-roadmap-20261010` (PR #15 head). The branch was
  synchronized with GitHub before editing this report; PR #15 was open and
  draft. No code or legacy artifacts were modified.

Commands:

```powershell
python scripts/extract_physicalai_action_pilot.py `
  --artifact-dir C:\CodexArtifacts\pretrained-reuse-r2\physicalai-stack-platforms-patha-fullpilot-20261102-attempt-01 `
  --model-dir C:\CodexArtifacts\pretrained-reuse-r2\dinov2-base-runtime-smoke-01\model `
  --device cuda --batch-size 4

$env:CUDA_VISIBLE_DEVICES=''
python scripts/evaluate_physicalai_action_pilot.py `
  --artifact-dir C:\CodexArtifacts\pretrained-reuse-r2\physicalai-stack-platforms-patha-fullpilot-20261102-attempt-01
```

The experiment output uses a new directory. Its 1,900 input entries are
read-only hard links to the already hash-verified source files plus a copied
receipt; the existing acquisition directory was not modified. No model or
media was added to Git.

## Results

Validation contains 1,000 queries from 250 episodes, four windows per
episode. Class supports for prototypes 0–7 are `78, 74, 106, 92, 69, 137,
88, 356`; the largest class is 35.6% of the set. The table reports sample
Accuracy, NLL, multiclass Brier, ECE-10, and unweighted mean per-class
Accuracy.

| Arm | Accuracy | NLL | Brier | ECE-10 | Macro class Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Proprio history | 70.4% | 5.4132 | 0.5910 | 0.0056 | 69.22% |
| Video, 8-frame mean | 58.4% | 7.3422 | 0.8273 | 0.0508 | 53.70% |
| Video, last frame | 50.6% | 8.5827 | 0.9823 | 0.0934 | 47.95% |
| Joint video + proprio | **72.5%** | **4.4898** | **0.5432** | 0.0771 | **71.26%** |
| Joint, episode-misaligned video | 54.8% | 7.6628 | 0.8884 | 0.1004 | 50.38% |
| Joint, missing video | 63.2% | 5.8972 | 0.7254 | 0.1135 | 62.44% |
| Joint, missing proprio | 32.1% | 11.0925 | 1.3232 | 0.2805 | 30.71% |
| Train-class prior | 35.6% | 1.8945 | 0.8124 | 0.0229 | 12.50% |

Each trained readout reached 100% train Accuracy with near-zero train loss.
For the joint model, the train-to-validation Accuracy gap is 27.5 percentage
points. Joint NLL and Brier are better than proprio-only, while ECE is worse
(7.71% versus 0.56%); these mixed metrics do not establish a calibrated
multimodal quality gain. Per-class joint Accuracy is
`65.4%, 91.9%, 78.3%, 69.6%, 63.8%, 70.8%, 54.5%, 75.8%` for the supports above.

Paired bootstrap uses 2,000 resamples of the 250 validation episode clusters,
with seed 17; all four windows from a sampled episode stay together.

| Joint minus comparison arm | Accuracy difference | 95% episode-bootstrap interval |
| --- | ---: | ---: |
| Proprio history (best single input) | +2.11 pp bootstrap mean (+2.1 pp observed) | −0.8 to +5.1 pp |
| 8-frame video mean | +14.07 pp | +10.6 to +17.7 pp |
| Last-frame video | +21.90 pp | +18.2 to +25.7 pp |
| Episode-misaligned video control | +17.62 pp | +14.2 to +21.2 pp |
| Missing-video control | +9.33 pp | +6.2 to +12.3 pp |
| Missing-proprio control | +40.32 pp | +36.7 to +43.9 pp |

Under the frozen decision rule, joint versus the strongest single-input arm
does not pass because its interval includes zero. Missing and shuffled controls
do show that predictions depend on video, and the missing-proprio control shows
strong dependence on state. This is evidence of input sensitivity, not proof
that video adds reliable quality beyond the best single input. Final disposition
is therefore **UNCERTAIN**, not a both-input pass.

## Runtime, artifact integrity, and reproduction

- Hardware: NVIDIA GeForce RTX 3080 Laptop GPU, 16 GiB. Python 3.11.9,
  PyTorch 2.6.0+cu124, CUDA 12.4, Transformers 5.6.2, PyAV 18.1.0,
  PyArrow 25.0.1, NumPy 2.4.6, Pillow 10.4.0; FP32, batch 4.
- Feature extraction completed with exit code 0 in 534.013 seconds reported by
  the runner (decode 180.299 seconds, DINOv2 inference 345.754 seconds). Peak
  PyTorch CUDA allocation was 402,263,040 bytes. The feature cache is
  28,569,116 bytes. The LBFGS evaluation ran on CPU with CUDA hidden and exited
  zero; the evaluator itself does not save a wall-time field, while the
  observed command session took about 70 seconds.
- `probe-checkpoints.pt` save/reload reproduced probabilities exactly (maximum
  absolute delta 0.0 for all four fitted arms).
- Independent verification recomputed all eight arms' metrics from the saved
  probability rows (maximum metric difference below `5e-6`, due to float32
  probabilities serialized then recomputed in float64) and reproduced all six
  paired bootstrap means and intervals exactly. All 1,000 prediction rows map
  to the declared validation IDs, every episode has windows 0–3, probability
  sums differ from 1 by at most `1.11e-7`, and no reserved ID is present.
- C: free space was 1.181 GiB before and 1.133 GiB after the full run and audit.
  No cloud compute was used and no new media download occurred in this run.

Local result directory:
`C:\CodexArtifacts\pretrained-reuse-r2\physicalai-stack-platforms-patha-fullpilot-20261102-attempt-01\`

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `features.npz` | 28,569,116 | `235660d6a8947f7c0a9908541e5415cc5db36149a334e6baa7d0d32b347d4ef9` |
| `feature-extraction.json` | 1,097 | `b481e7ca31da2abecfce610042793ad716ff2dc9faef5fe330122abb73d9e39e` |
| `pilot-metrics.json` | 15,846 | `9e8aff98862a439b36722026e521ad0be458e30d64d19b8b104a237be49819e0` |
| `validation-predictions.json` | 2,021,438 | `3f69c09963c6515e138b581d35be66002b0362ac770391a6aed7b7c6bf8e21a2` |
| `probe-checkpoints.pt` | 150,595 | `d8e97f5040001e365b6031a97b471aabb483082cb1284d91d98fc1f3165b9f72` |
| `prediction-integrity-audit.json` | 3,477 | `d0d966e8ea39e50c8a669c53844ff6d04cbd0a25a1024ebc4fe39e7b25017d3f` |
| `run-metadata.json` | 7,559 | `871ba9cbad2bfa9639c6a0369f3cb1676b5879b419139562c4b1c4d1976f838b` |

The machine-readable summary is
[`physicalai-stack-platforms-fullpilot-result-20261102.json`](../../manifests/pretrained_reuse/physicalai-stack-platforms-fullpilot-result-20261102.json).

## Next gate

Do not rerun or tune on this validation split. Before any product-directed
PhysicalAI follow-up, obtain scene/layout identity or a genuinely independent
scene-held source, define a deployment-aligned action task, and establish an
untouched evaluation set. A separate source/task must still show a reliable
joint-over-best-single gain with acceptable calibration. This result does not
close R2-M, does not justify R3 shared-backbone or R4 alignment work, and does
not satisfy the cross-modalities or product-quality goals.

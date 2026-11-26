# PhysicalAI pilot metric and reproduction corrigendum — 2026-11-02

## Disposition

The 700-train / 250-validation PhysicalAI stack-platforms pilot was completed
and its validation was exposed in the original run documented in
[`R2_PHYSICALAI_STACK_PLATFORMS_ACTION_PILOT_20261102.md`](R2_PHYSICALAI_STACK_PLATFORMS_ACTION_PILOT_20261102.md).
A later run, recorded in
[`R2_PHYSICALAI_STACK_PLATFORMS_FULLPILOT_RESULTS_20261102.md`](R2_PHYSICALAI_STACK_PLATFORMS_FULLPILOT_RESULTS_20261102.md),
was an accidental deterministic reproduction after the original report had
already been committed. It is **not an independent run or new quality
evidence**. Do not count it as a second seed, a fresh validation, or evidence
that reduces uncertainty.

The original and reproduction have matching SHA-256 hashes for all three
primary outputs:

| Artifact | Original and reproduction SHA-256 |
|---|---|
| `features.npz` | `235660d6a8947f7c0a9908541e5415cc5db36149a334e6baa7d0d32b347d4ef9` |
| `validation-predictions.json` | `3f69c09963c6515e138b581d35be66002b0362ac770391a6aed7b7c6bf8e21a2` |
| `probe-checkpoints.pt` | `d8e97f5040001e365b6031a97b471aabb483082cb1284d91d98fc1f3165b9f72` |

The prediction hash above is the exact SHA-256 stored in both output
directories and the audit JSON. The original run started at 2026-11-02
14:47:50 JST. The reproduction's exact start time is not preserved and is
`UNKNOWN`. Both runs used the same fixed split, task, seed, encoder revision,
and train/validation IDs.

## ECE correction

Both historical PhysicalAI metric implementations used a half-open final bin
ending at `1.0`. Consequently, rows whose maximum class probability was
exactly `1.0` were omitted. The saved probabilities are often exactly one due
to FP32 softmax saturation. This made the previous ECE values substantially
too low. The values below are computed from the immutable original saved
prediction rows with ten equal-width bins, including confidence `1.0` in the
last bin. No model, encoder, validation features, or labels were reloaded;
there was no inference, fitting, optimizer update, checkpoint selection, or
sealed-audit access.

| Arm | Accuracy | NLL | Brier | Previously reported ECE-10 | Corrected ECE-10 | Rows at confidence 1.0 (excluded before) |
|---|---:|---:|---:|---:|---:|---:|
| Proprio history | 70.4% | 5.4132 | 0.5910 | 0.0056 | **0.2956** | 990 / 1,000 |
| Video, 8-frame mean | 58.4% | 7.3422 | 0.8273 | 0.0508 | **0.4128** | 917 / 1,000 |
| Video, last frame | 50.6% | 8.5827 | 0.9823 | 0.0934 | **0.4904** | 857 / 1,000 |
| Joint video + proprio | 72.5% | 4.4898 | 0.5432 | 0.0771 | **0.2681** | 831 / 1,000 |
| Joint, episode-misaligned video | 54.8% | 7.6628 | 0.8884 | 0.1004 | **0.4384** | 816 / 1,000 |
| Joint, missing video | 63.2% | 5.8972 | 0.7254 | 0.1135 | **0.3585** | 790 / 1,000 |
| Joint, missing proprio | 32.1% | 11.0925 | 1.3232 | 0.2805 | **0.6585** | 600 / 1,000 |
| Train-class prior | 35.6% | 1.8945 | 0.8124 | 0.0229 | **0.0229** | 0 / 1,000 |

Accuracy, NLL, Brier, class support, saved predictions, checkpoints, and the
paired episode bootstrap are unchanged. The corrected ECE confirms substantial
overconfidence across all fitted arms; the joint model's 0.2681 ECE is not a
calibration pass. The joint-minus-proprio Accuracy interval remains
`[-0.8, +5.1]` percentage points and includes zero. The pilot remains
`UNCERTAIN` / `REVIEW_NOT_ADMITTED` for the already documented scene/layout,
task-alignment, and generalization limitations.

### Affected fixed-checkpoint full development validation

The same metric implementation scored the 908-query / 227-episode fixed-
checkpoint development validation in
[`R2_PHYSICALAI_FULL_VALIDATION_20261102.md`](R2_PHYSICALAI_FULL_VALIDATION_20261102.md).
Its predictions and result JSON were also corrected from saved outputs only;
the 64-episode pilot checkpoint was not reloaded and no inference or fitting
was performed.

| Arm | Previously reported ECE-10 | Corrected ECE-10 | Rows at confidence 1.0 (excluded before) |
|---|---:|---:|---:|
| Proprioception history | 0.0227 | **0.1890** | 861 / 908 |
| Video, 8-frame mean | 0.1762 | **0.1839** | 340 / 908 |
| Video, last frame | 0.1678 | **0.1700** | 245 / 908 |
| Joint video + proprioception | 0.1618 | **0.1629** | 341 / 908 |
| Joint, video replaced by train mean | 0.2209 | **0.2209** | 6 / 908 |
| Joint, proprioception replaced by train mean | 0.2066 | **0.2077** | 136 / 908 |
| Joint, episode-misaligned video | 0.3402 | **0.4558** | 239 / 908 |
| Train class prior | 0.0113 | **0.0113** | 0 / 908 |

The fixed-checkpoint joint ECE remains 0.1629 and proprio ECE is 0.1890. This
corrects calibration reporting; it does not change the episode-paired Accuracy
interval, the exposed-validation status, or the source's `REVIEW_NOT_ADMITTED`
disposition. Full-validation audit output SHA-256 is
`74c8f58de85cf686f8b615abd35968b884cfe2398dd5934af71c138191209e76`; its
input prediction SHA-256 is
`103958268b9283f8faa5ccae0ae6d099c3f5df8ec04381c266635c8c658e0995` and its
historical result JSON SHA-256 is
`6985057d9703c3ebe6cd6da03456d333089c83544dfd0c525733ada0bd0878b4`.
The audit is stored in
`C:\CodexArtifacts\pretrained-reuse-r2\physicalai-full-validation-ece-correction-20261102-attempt-01\`.

## Code and reproducibility

- Added `expected_calibration_error` with confidence `1.0` assigned to the last
  bin; both PhysicalAI pilot and full-validation evaluators now share it.
- Added regression coverage for confidence exactly `1.0` and ordinary bin
  boundaries. Targeted tests: 7 passed. Ruff and `git diff --check` passed.
- Saved-prediction audit script:
  `scripts/audit_physicalai_saved_prediction_metrics.py`.
- Audit output directory:
  `C:\CodexArtifacts\pretrained-reuse-r2\physicalai-pilot-ece-correction-20261102-attempt-02\`.
- Audit output `result.json` SHA-256:
  `b297f417723117e906d477d0f557cfb0eee29fd9980aa92e4dc1f592cd5029b9`.
- Audit script SHA-256:
  `04a19de90d2d83a90f3bf13e4ae30a4209e82dbc7edc91a7f389d63636d6790a`.
- Original predictions SHA-256:
  `3f69c09963c6515e138b581d35be66002b0362ac770391a6aed7b7c6bf8e21a2`.
- Original `pilot-metrics.json` SHA-256:
  `e7ce5b7787ac27f7d3809f9d7fb3191368a5292fe115b3c04d819e852b517694`.
- Full-validation predictions and result JSON were audited separately from
  their saved outputs; no reserved data or sealed audit was opened.
- Reproduction `run-metadata.json` SHA-256:
  `871ba9cbad2bfa9639c6a0369f3cb1676b5879b419139562c4b1c4d1976f838b`.
- The reproduction began with 1.181 GiB free on C:, below the R0 policy's
  2 GiB minimum reserve. This was a process violation. Do not start further
  expensive runs until the reserve is restored.

The original report, original and reproduction prediction/feature/checkpoint
files, both `pilot-metrics.json` files, and both run directories remain
preserved. The machine-readable correction is
[`physicalai-pilot-ece-correction-20261102.json`](../../manifests/pretrained_reuse/physicalai-pilot-ece-correction-20261102.json).

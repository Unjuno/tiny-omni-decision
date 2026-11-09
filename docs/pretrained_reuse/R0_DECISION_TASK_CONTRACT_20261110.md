# R0 supplied-option Decision contract — 2026-11-10

## Scope and status

This freezes the engineering interface and evidence rules for the existing
Path A family. It does **not** freeze a user-approved release task battery,
select an end-user device, or qualify the current model. The machine-readable
contract is
[`configs/pretrained_reuse/decision_task_contract_v0_1.json`](../../configs/pretrained_reuse/decision_task_contract_v0_1.json).

The product accepts text, image, audio, and video as required input routes;
sensor remains optional. Requests carry a caller-owned request ID, an explicit
criterion, and at least two supplied options with unique stable IDs and text.
The scorer must preserve option order and return one probability per option.
Ordinal decisions additionally require a finite, strictly increasing numeric
value for each option. Audio/video tasks record the time segment, frame order,
timestamps, and clock reference whenever temporal alignment matters. Joint
requests additionally pin each asset and record alignment and missingness.

## Output semantics

The implementation reference is
[`decision_json.py`](../../src/tiny_omni_decision/decision_json.py). Its JSON
contains `schema_version`, `request_id`, `decision_type`,
`selected_option_id`, `confidence`, and the ordered option records. Probabilities
must be finite, lie in `[0,1]`, and sum to one within `1e-5`; the selected ID
must point to a maximum-probability option. An explicit abstention form is
supported. Evaluation counts abstentions as wrong in all-query accuracy and
reports coverage separately.

This schema validates a probability distribution; it does not calibrate it.
Calibration must be measured per task using NLL, Brier, ECE-15, confidence, and
reliability bins under the existing source-group bootstrap protocol.

## Per-modality input contracts

| Route | Required evidence in a request | Current task status |
|---|---|---|
| Text | Explicit state/text, criterion, and supplied options; target labels must not leak through the context. | Typed Decisions Synth is a synthetic development baseline only. |
| Image | Pinned image identity/hash, criterion/options, and a recorded crop/resize policy. | CLEVR-4 is synthetic; it does not establish general image reasoning. |
| Audio | Pinned audio asset, sample rate/channels, criterion/options, and segment/clock metadata when applicable. | Speech Commands is a closed-set keyword task, not general speech understanding. |
| Video | Pinned video identity/hash, criterion/options, frame order and sampling/timestamp policy. | CLEVRER is development evidence; its temporal quality gate remains unmet. |
| Sensor (optional) | Timestamped channels with units and explicit validity/missingness. | No sensor task is in the required first battery. |

The current known tasks are retained as development baselines. The manifest
leaves each release task ID unset and the final battery open. No user-facing
task is invented to make R0 appear complete.

## Evaluation gates

The versioned manifest pins the existing policy file by SHA-256. Each named
task needs at least 1,000 examples and 200 independent source groups. The
provisional target is 95% accuracy with a 95% group-bootstrap lower endpoint
of at least 95%; 98% remains a stretch target. Report Accuracy, NLL, Brier,
ECE-15, mean confidence, coverage, reliability bins, and the effective group
count separately for every task. The policy retains 10,000 paired group
bootstrap resamples with seed 17.

A joint-input task is not admitted merely because two media streams are
present. Before fitting, freeze best-unimodal, missing-input, shuffled-input,
stale/time-offset, and contradiction controls. The fused model must improve
both Accuracy and NLL over the best unimodal control under paired source-group
uncertainty. No current fresh paired-source candidate satisfies all rights,
group-count, and task-necessity gates.

## Open decisions

- End-user deployment device beyond the confirmed engineering reference.
- Latency and energy ceilings for that device.
- A user-approved task for each required modality and a fresh locked evaluation
  cohort for each task.
- A rights-cleared joint-input task with adequate independent groups.

The local Windows RTX 3080 (16 GiB), CPU fallback, no-cloud rule, 100–200M
pre-quantization parameter aspiration, and per-task 95% accuracy target remain
the engineering reference. The currently measured Path A bundle is
170,499,716 unique parameters and 687,254,772 bytes; the provisional 200 MB
package target remains unmet. The contract does not change or promote existing
checkpoints, corpora, results, or artifacts.

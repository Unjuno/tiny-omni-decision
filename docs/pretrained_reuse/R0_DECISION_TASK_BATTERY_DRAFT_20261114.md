# R0 Decision task battery — engineering draft (2026-11-14)

Status: **proposed engineering contract; not a product release approval**.
This draft narrows source search and Path A experiments to one coherent product
behavior. It does not claim that any current benchmark passes the contract.

## Product behavior

Given an observation (text, image, audio, video, or an explicitly declared
combination), a task instruction/criterion, and caller-supplied candidate
options, return a calibrated probability for each supplied option in a typed
JSON response. The model must preserve option IDs and order semantics, expose
abstention when the evidence is missing or out of scope, and never invent a new
candidate. Text may describe the task or criteria, but it must not be a hidden
label channel when the task claims to be image-, audio-, video-, or sensor-led.

Example response contract (field names subject to the existing canonical
runtime schema):

```json
{
  "decision_id": "caller-id",
  "probabilities": [
    {"option_id": "candidate-a", "probability": 0.72},
    {"option_id": "candidate-b", "probability": 0.28}
  ],
  "abstain": false
}
```

Probabilities must be finite, normalized, and keyed to the exact supplied
option IDs. Missing, stale, contradictory, or unsupported evidence must follow
an explicit abstention policy instead of silently falling back to another
modality.

## Required task families

Each family is evaluated separately. These are task requirements, not
benchmark endorsements; a source must independently pass rights, exposure,
grouping, and sample-count gates before its data are used.

| Track | Decision task | Input requirement and shortcut controls |
| --- | --- | --- |
| Text | Select among supplied actions/answers using the stated facts and criteria. | Counterbalance option IDs/order; include unseen source/state groups; question-only and option-prior controls. |
| Image | Select a supplied answer/action from visible scene evidence (attributes, counts, relations, or state). | Text-only, position/prior, blank-image, and matched image-shuffle controls; require scene/object-grouped splits. |
| Audio | Select a supplied answer/action grounded in the audio signal, including semantically meaningful speech or sound. | Audio-only capability must be separated from transcript/text controls; use speaker/recording-group splits and hard negatives where appropriate. |
| Video | Select a supplied answer/action that depends on temporal state, order, or change. | Compare static-frame, reversed, shuffled, and question-only controls; split by source video/scene/session. |
| Optional sensor | Select a supplied action/state from typed temporal sensor observations. | Preserve units, timestamps, missingness and subject/session grouping; do not make this a required product modality until a task passes. |
| Joint multimodal | At least one AV or video/sensor task whose correct option cannot be determined from either stream alone. | Freeze best-unimodal, each single-stream ablation, missing, shuffled, stale/time-offset, and contradiction controls before fitting. The fused model must beat the best unimodal control on Accuracy and NLL with paired group-level uncertainty. |

For the joint task, the preferred product pattern is **evidence-conditioned
action/option selection**: for example, audio supplies a spoken reference or
event and video supplies the scene/entity/state needed to resolve the option.
This is a task hypothesis only. Do not construct labels or claim complementarity
until a rights-cleared source and independent groups support it.

## Data and evaluation gates

- At least 1,000 independent examples and 200 independent source groups per
  named task are the provisional minimum; smaller samples may be reported as
  diagnostics but cannot pass the quality gate.
- Define group identity before inspecting evaluation outcomes. Examples from
  the same speaker, scene/video, image/source state, recording session, or
  derived parent event must not cross splits when that identity can induce
  leakage.
- Keep train, calibration, validation, and sealed evaluation roles explicit.
  The sealed set is opened once after architecture, sampling, checkpoint rule,
  and calibration are frozen.
- Report per-task Accuracy, macro-by-group Accuracy, NLL, multiclass Brier,
  ECE-15, mean confidence, coverage/abstention, reliability bins, and
  source-group paired 95% intervals. Do not pool unrelated tasks into a pass.
- Retain the existing provisional rule: 95% Accuracy point estimate and lower
  two-sided 95% source-group bootstrap endpoint at least 95%; 98% remains a
  stretch target. ECE-15 point and upper interval <=0.05; NLL/Brier must beat
  uniform-option and train-only majority baselines under paired group
  uncertainty. Calibrators fit only on the separate calibration groups.
- For a joint task, require the fused candidate to improve both Accuracy and
  NLL over the best unimodal control under paired group-level uncertainty;
  otherwise mark joint-input necessity `UNCERTAIN` or `FAIL`, not a fusion win.

## Model and runtime envelope

- Count every unique resident encoder, tokenizer/embedding, bridge, fusion
  module, temporal state, and Decision head before quantization. Engineering
  target: 100–200M total unique pre-quantization parameters.
- The <=200 MB package target applies only to an actual quantized export whose
  full artifacts are counted and whose save/reload probabilities pass parity.
  It is not a parameter-count limit and is not established by theoretical
  bit-width arithmetic.
- Engineering reference: the user's local Windows PC (i7-12700H, 31.75 GiB
  RAM, RTX 3080 Laptop GPU 16 GiB, Intel Iris Xe). CPU fallback is required to
  be measured. No paid/cloud compute.
- Current local hardware is a **development reference**, not yet a confirmed
  end-user deployment device. Record p50/p95 end-to-end and decision-only
  latency, peak RAM/VRAM, energy/power where measurable, and cold-start separately.
  Numerical latency/energy release ceilings remain open; do not invent a pass
  threshold from a single machine snapshot.

## What is and is not admitted now

- PR #94's four-route typed JSON and parity evidence is runtime plumbing only;
  its fixed development inputs are not a quality benchmark.
- MiniWoB's fresh-seed score is not an image-quality pass because a deterministic
  RGB rule solves the criterion-conditioned cohort.
- CATER's observed validation shows no video dependence over question-only;
  do not tune on that cohort again.
- The old Speech Commands model was near chance on LibriSpeech transcript
  selection with either random or phoneme/lexical/length-matched distractors;
  native Whisper nearly saturated both conditions. Distractor hardness alone
  therefore does not establish broad audio Decision quality.
- TartanAviation has 126 aircraft-linked video components (6 under
  day-plus-aircraft grouping); it fails the provisional 200-group gate.
- Localized Narratives and RoboMIND2 remain unadmitted under their current
  rights/group/task evidence.

No current source is admitted for the joint multimodal task. The next R2 source
screen must map an official, product-compatible dataset to a specific row of
the required-task table and demonstrate group/sample feasibility before any
payload acquisition, feature extraction, fit, or evaluation.

## Decisions still required before release freeze

The task behavior, per-task quality/calibration gates, local engineering
reference, and no-cloud policy above are proposed for forward-looking
development. The eventual end-user device, latency/energy ceiling, and final
approved task battery remain open in R0. Product promotion and R8 locked
evaluation remain blocked until those decisions and clean task data exist.

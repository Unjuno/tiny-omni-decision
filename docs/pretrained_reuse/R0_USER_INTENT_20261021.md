# R0 user-intent clarification — 2026-10-21

This dated supplement records the user's later clarification of the product
goal. It does not change old runs, their thresholds, or their measured results.

## Confirmed engineering intent

- The complete inference system should contain no more than 200 million
  **pre-quantization parameters**. Quantization changes storage and arithmetic
  representation, not this parameter-count target.
- Count every resident encoder, tokenizer/embedding table, modality adapter,
  fusion module, and decision readout. Report the quantized package size
  separately; the roadmap's provisional 200 MB package target remains a
  separate engineering measure, not a parameter-count proxy.
- Return custom JSON containing probabilities over the caller-supplied typed
  options. The model need not produce long-form natural-language answers.
- The desired system accepts text, image, audio, and video decision inputs.
  Sensor input is optional. Report each required modality and any true joint
  input task separately; do not let a strong modality hide a weak one in a
  pooled score.
- The product goal is at least 95% Accuracy for each predeclared required
  task; 98% is an aspiration. These are goals, not results or an accepted
  release gate until task definitions, independent test groups, uncertainty,
  calibration, and abstention policy are frozen.
- Temporal/recurrent designs such as LSTM or xLSTM may be compared where the
  task depends on sequence order. They are candidates, not mandatory
  architecture choices; adopt only with matched task evidence and full-system
  parameter/runtime accounting.

## Still open

- Deployment device is not specified. The local RTX 3080 Laptop GPU with
  16 GiB VRAM is an engineering/measurement host only; do not treat it as the
  end-user target without confirmation.
- The per-modality task battery and any mandatory joint AV/other task are not
  frozen. Consequently, exact task-specific release floors, confidence-bound
  requirements, calibration thresholds, and abstention thresholds remain
  unset.
- No numeric latency, energy, wall-time, or storage budget beyond the
  provisional package-size target has been supplied. Measure and report these
  on the engineering host; do not invent a universal product limit.
- Compute remains local-only with no paid cloud spend. Use the cooperative
  HotUGPU guard for any GPU workload; it must wait for a safe reservation and
  must not terminate or signal the existing supervisor/workload.

## Consequence for R0–R8

This clarification is enough to guide bounded engineering experiments and
architecture comparisons. It is not enough to select or promote a product
model. Continue R1/R2 with pinned, task-specific evidence; preserve separate
unimodal and joint-task results; and defer release qualification until the
open device and evaluation-contract items are resolved.

The old R0 snapshot remains a record of what was unknown when it was captured.
This supplement is the later user-intent update; neither record supersedes
historical experiment evidence.

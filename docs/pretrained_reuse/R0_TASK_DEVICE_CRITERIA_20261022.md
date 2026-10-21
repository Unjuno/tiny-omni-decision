# R0 research device and quality criteria — 2026-10-22

## Status

This records provisional **research engineering** criteria from the user's
product objective and the hardware actually present. It does not establish a
product deployment target, admit any dataset, or authorize promotion. The
deployment device and application-specific risk/cost profile remain unknown,
so release qualification remains open.

## Reference hardware and compute policy

- Local research reference: Windows PC with NVIDIA GeForce RTX 3080 Laptop GPU,
  16,384 MiB, WDDM driver 616.92. CPU-only execution is also measured when a
  pinned encoder requires it. This is the benchmark host, not a claim that the
  shipping device is a laptop or has this GPU.
- All GPU work must use the repository's cooperative HotUGPU wrapper and must
  leave existing processes untouched. No cloud GPU or hardware purchase is
  authorized. Cloud spend target is $0; local electricity cost is unmeasured.
- Record wall time, GPU/CPU, peak allocated and reserved memory, and batch-one
  cold-start and steady-state latency. No latency ceiling is set until the
  actual product device and decision deadline are named.

## Model size and output

- Engineering model-size target: 100–200M **underlying pre-quantization
  parameters**. Count every unique parameter resident for the selected task,
  including encoders, adapters/lenses, fusion, readout, and optional sensor
  paths. Quantization changes storage/compute representation, not the original
  dense parameter count.
- Initial serialized-package target: at most 200 MB including weights,
  tokenizers, scales, exceptions and runtime assets. This remains a provisional
  engineering target and must be measured after export.
- Output is a versioned JSON decision object containing caller-supplied option
  IDs and normalized probabilities in input order. Validate schema, finite
  values, unique IDs, and probability sum tolerance `1e-6`.

## Quality battery and provisional thresholds

The required modalities are evaluated separately: text, image, audio, and
video. Joint audio-video (or another genuinely multi-input task) is reported as
its own task with matched unimodal controls. Sensor is optional and, if used,
gets its own task and inclusion in parameter/package accounting.

- Accuracy challenge floor: at least **95% on each predeclared task**; 98% is
  the stretch target. Never substitute an aggregate or a strong modality for a
  failing task.
- Calibration: report NLL, Brier, and 15-bin ECE with reliability counts for
  every task. Provisional calibration screen is ECE-15 `<= 0.05`, alongside no
  material NLL/Brier regression against that task's frozen reference. This is
  an engineering screen, not a safety-certified confidence guarantee; a real
  use case may require stricter risk-specific calibration and abstention rules.
- Use group-disjoint train/validation/evaluation IDs; retain source/task and
  subgroup metrics. Select checkpoints and settings on validation only. Keep
  the final audit sealed until architecture, data, selection and thresholds
  are frozen.
- For any fusion claim, require paired joint-versus-best-unimodal results and
  valid stream interventions. An input-replacement sensitivity change alone
  does not prove correctness or complementarity.

## Open R0 items

The product device, task-specific application costs, decision deadline,
abstention policy, and safety consequence are still unspecified. Until those
are established, the thresholds above guide research comparisons only; no
latency, power, or release PASS can be claimed. Existing narrow benchmarks do
not yet satisfy the four-modality 95–98% objective.

# Path A four-route JSONL runtime follow-up — 2026-11-10

## Result

The existing task-scoped Path A dispatcher returned finite typed option
probabilities for Text, Audio, Image and Video requests on the local CPU. The
routes were exercised separately. This does **not** show that the model fuses
multiple modalities, that any score is accurate or calibrated, or that the
release task battery is met.

| Route | Fixed input used | Output | Runtime observation |
|---|---|---|---|
| Text | Synthetic red-object sentence; `[red, blue]` options | JSON, `P(red)=0.5662` | 4.72 s dispatch including worker startup; 0.036 s request |
| Audio | Generated one-second 440 Hz tone; two query templates | Two JSON records; max probabilities 0.7201 and 0.8775 | 0.150 s request after shared worker startup; tone has no spoken command or quality label |
| Image | Generated PNG; VideoMAE V2 static-image route | JSON, `P(red)=0.7569` | 4.06 s dispatch including worker startup; 0.422 s request |
| Video | One MIT-IBM CLEVRER **train** clip (`video_00000.mp4`, CC0-1.0), one supplied binary candidate | JSON, `P(correct)=0.5017` | 16.60 s dispatch including worker startup; VideoPrism forward 7.16 s; reported peak working set 4.16 GB |

These are protocol and inference-path observations only. No Accuracy, NLL,
Brier, ECE, validation, checkpoint selection, training, or sealed-audit result
was computed. The generated Text/Audio/Image inputs are not evaluation items.
The Video request used a training asset and was not scored against its target.

## Why the first attempt failed

The previous smoke log was interpreted as possibly indicating a Windows DLL
application-control block. The isolated rerun narrows that diagnosis:

- The first launcher exposed the existing workspace `.venv` `site-packages`
  through `PYTHONPATH`; the pinned VideoPrism Python environment then imported
  the caller's incompatible CPU Torch/torchvision instead of its own packages.
- The new Path A Torch worker imports `psutil`, but the project ML extra did not
  declare it. Text and Audio therefore still failed on the first isolated run.
- A clean dedicated Torch environment, a source-only `PYTHONPATH` for the
  separate VideoPrism process, and a pinned `psutil` installation made all four
  routes run. VideoPrism used JAX's CPU backend. No application-control bypass
  was attempted, and a system-wide policy explanation is unsupported by this
  rerun.

`pyproject.toml` now declares `psutil>=5.9,<8` in the optional `ml` extra so a
normal `.[ml]` installation supplies the worker dependency.

## Frozen resources and budget

- Path A code commit: `e33a0bf0` (`codex/path-a-videomae-image-jsonl-runtime-20261110`).
- Roadmap base: `e07e049ff4387484fb754e0fff963631f3cc7c85`.
- Runtime: CPython 3.11.9, Torch 2.6.0+cu124, Transformers 5.6.2, PyAV 18.1.0,
  psutil 7.2.2; RTX 3080 Laptop present, but all requests used `--device cpu`.
- VideoPrism environment: CPython 3.11.16, JAX 0.6.2, TensorFlow 2.18.1,
  `JAX_PLATFORMS=cpu`.
- RTX 3080 was 0% / 0 MiB after the smoke. No training or GPU inference ran.
- Clean v4 assembly manifest SHA-256:
  `79fbca05e6062b66fe6ffb8a831d6c635edffe361c76ea4ce1574f0736ae9519`.
  Its unique parameter union is 170,499,716 and its files total 687,254,772
  bytes excluding the manifest. It declares no joint Audio/Video capability.
- The separate VideoMAE image route adds 22,205,968 parameters (encoder plus
  readout; encoder checkpoint 44,334,609 bytes; readout 212,985 bytes). A
  simple combined count is 192,705,684 parameters and about 731,802,366 bytes
  before any quantization, package metadata or runtime dependencies. The count
  remains under the provisional 200M parameter ceiling; the package does not
  meet the provisional 200 MB target.
- All model artifacts and the clean assembly were read-only. The JSONL outputs
  and logs are external attempt artifacts under
  `D:\CodexArtifacts\pretrained-reuse-r2\path-a-runtime-*`.

The `VideoMAE` image route repeats a static image across 16 frames and reports
`temporal_evidence=false`. Video uses a separate VideoPrism encoder and readout.
The four-route interface is therefore a packaging/API integration across
task-specific towers, not one shared Omni representation. No multimodal
quality or modality-necessity gate is closed.

## Reproduction records

All requests were sent through:

```text
python -B -m tiny_omni_decision.cli path-a-jsonl --assembly <clean-v4-copy> \
  --videoprism-python <pinned-videoprism-python> \
  --videoprism-source <pinned-videoprism-source> \
  --videomae-model-dir <pinned-videomae-checkpoint-dir> \
  --videomae-readout <best-readout.pt> \
  --videomae-dependency-dir <pinned-videomae-dependencies> \
  --worker-log-dir <new-empty-output>/worker-logs --device cpu
```

For the VideoPrism child process, `PYTHONPATH` contained source roots only;
`JAX_PLATFORMS=cpu` selected CPU execution. The successful attempts are
immutable and hash-addressed:

| Artifact | SHA-256 |
|---|---|
| Attempt 05 Text/Audio/Image requests | `bf74a021a810b6c38f375381abfe5693538d4d283c9348f6418f2d6b5404da4f` |
| Attempt 05 Text/Audio/Image responses | `7b7b63742ca1a1fc352395c0c5fd15e89be1492e3a094f2249ce6f988815d417` |
| Attempt 06 Video request | `9fc44f83f0c9bb1041d89a57e2743c00b8e52aba41b3fee5b60d5b99ace67c6f` |
| Attempt 06 Video response | `494813f88ed43534e3123647468921ddf867d1eb7112446b636785ee88f753b3` |
| Attempt 06 Torch worker log | `701fc3e0a46ddd9534c517fe572c26208290fb97f63bfe35fb11cd7c366fd0f9` |
| Video train asset | `0fa71e14a9f5d1852bbba270808037ee29f329090189c38dea11844e2b31c060` |
| VideoMAE readout | `437e920e7cda02ff2e28f19c96433af04aab3d70494030b6e3278809aa1cc39e` |
| VideoMAE encoder weights | `24fb71687fa3671b8387cadfbcbab0f72af695692e93cf1ecc82caa888626172` |

Attempts 01–04 and their errors remain preserved beside the successful
attempts; they were not overwritten. The old report records the initial
failure; this follow-up corrects its cause based on the isolated rerun.

## Code verification

- Ruff on the changed dependency contract and Path A runtime test files: pass.
- Focused CPU tests for the dependency contract, JSONL protocol, VideoMAE
  runtime and frozen Decision task contract: **12 passed**.
- The runtime smoke is an additional host integration check; it is not counted
  as a model-quality test.

## Next gate

The next quality work is still R2-M: admit a rights-cleared, group-split task
whose target requires multiple input streams, then predeclare matched
unimodal/fusion and perturbation controls. Do not spend the exposed validation
IDs or interpret this runtime smoke as task performance. In parallel, package
and parameter reductions must be measured without claiming ternary compression
until the model is actually quantized and reload-verified.

# Path A JSONL multimodal runtime smoke — 2026-11-10

## Result

A CPU-only JSONL smoke was sent through the task-scoped Path A dispatcher for
one request per modality. Text and Audio returned valid typed Decision JSON.
Image and Video returned JSON error records because the shared VideoPrism
worker could not import TensorFlow during startup. The failure occurred before
image/video decode or encoder forward; those two routes are therefore **not
verified end to end on this Windows environment**.

| Route | Input | Result | Interpretation |
|---|---|---|---|
| Text | Generated sentence and two color choices | `ok`, normalized typed output | Protocol/load smoke only |
| Audio | Generated one-second 440 Hz mono PCM tone, one two-option query | `ok`, normalized typed output | Protocol/load smoke only; tone contains no spoken word or quality label |
| Image | Generated PNG | `OSError: [Errno 22] Invalid argument` in JSON response | VideoPrism worker startup failed before decode |
| Video | Generated eight-frame MP4 | Same error | Same worker startup failure before decode |

The VideoPrism worker log ends at `ImportError: DLL load failed while
importing tensorflow.python._pywrap_sanitizers`. Its localized Windows error
text is consistent with an application-control DLL block. The exact policy
rule and whether another already-approved runtime profile works are UNKNOWN.
No attempt was made to bypass that policy or replace the pinned environment.

## Conditions and evidence

- Source commit: `4fe0a51d519417713b56feb829a6025313d278c5`.
- Device: `--device cpu`; `CUDA_VISIBLE_DEVICES` was empty. The RTX 3080 was
  observed at 0 MiB / 0% during the run.
- Assembly: preserved clean v4 copy, manifest SHA-256
  `79fbca05e6062b66fe6ffb8a831d6c635edffe361c76ea4ce1574f0736ae9519`; the
  dispatcher accepted its manifest before worker startup. The assembly itself
  was not modified.
- Request JSONL SHA-256:
  `3f8362ddb662ee4ceb43109818c2fc889659da3246126563fdd6a040a63d4e32`.
- Response JSONL SHA-256:
  `1b4bb7c6bcb2ee159fb3688797fbdd8c78b8687236cb099f553fa73c6d13c23b`.
- VideoPrism worker log SHA-256:
  `b43a6c6835910459cf232c54ff16647570e3f77af2ec589c0cf7fa64ad8f6be2`.
- Full request, response, generated synthetic media, worker logs and receipt
  are preserved under
  `C:\CodexArtifacts\pretrained-reuse-r2\path-a-jsonl-four-modality-smoke-20261110-attempt-01\`.

No public dataset, validation corpus, training data, checkpoint selection or
sealed audit was loaded. There was no training. The two successful outputs are
not accuracy, calibration, generalization or latency evidence. The Text and
Audio scores are incidental outputs on synthetic inputs.

## Next runtime gate

Resolve the pinned VideoPrism TensorFlow DLL startup failure only through an
approved existing environment/runtime. Then repeat a single synthetic Image
and Video request with the same hash-checking JSONL path and a new output
folder. Until that passes, the JSONL interface is executable for Text/Audio
only in the tested local environment; Image/Video routing remains code-level,
not host-verified. Product task quality, joint-input evidence, release task
battery, package size and device limits remain open.

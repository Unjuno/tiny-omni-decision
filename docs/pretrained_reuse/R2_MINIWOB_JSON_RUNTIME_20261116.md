# MiniWoB DINOv2 Decision JSON Runtime Parity (2026-11-16)

## Scope

This change exposes the already selected MiniWoB screen readout through the
repository's `classification_json` contract and replays the existing frozen
development predictions through it. This is a serialization/runtime integrity
check, not a new quality evaluation, checkpoint selection, calibration fit, or
training run. The old Path A results remain the quality record.

The scorer consumes cached option features shaped `[1, 4, 1154]`; it does not
itself load DINOv2 or decode screenshots. End-to-end image encoding remains a
separate runtime integration task. Probabilities use temperature 1.0. No
calibration was fitted.

## Immutable inputs and execution

- Source branch base: `docs/pretrained-reuse-roadmap-20261010` at
  `bfb13bf6e9ac4f2ce244eb51e470afa907587380`.
- Frozen run: `D:\CodexArtifacts\pretrained-reuse-r2\miniwob-dino-path-a-v1-attempt-01`.
- Frozen corpus: `D:\CodexArtifacts\pretrained-reuse-r2\miniwob-screen-only-corpus-20261112-attempt-02`.
- Selected checkpoint SHA-256:
  `983366413dc2afd9f21a5411dd8793525f68342c44936d3bb2ccffe123f5c3fb`.
- Effective config SHA-256:
  `97fc8827ec0f322800560d9853fd10304d90bb414ae2ae14cc0e942b23a737a3`.
- Frozen feature cache SHA-256:
  `051ff80289b2b5a3232db524486bed8f6b939938a24358707758e4d62fe7eaef`.
- Replay script SHA-256:
  `57025c8fb1fc5e47e10bdabc8b5b70315ef5f71f706e37c5e54a49d6e5461c50`.
- Validation JSONL SHA-256:
  `2be09a9f7f9a4d85f9d642f798b69d2fc39d5fcac0b0a0f9993d4fad892a350f`.
- Command:
  `python scripts/replay_miniwob_json_runtime.py --run-dir D:\CodexArtifacts\pretrained-reuse-r2\miniwob-dino-path-a-v1-attempt-01 --output-dir C:\CodexLocalArtifacts\pretrained-reuse-r2\miniwob-json-runtime-20261116-attempt-01`
- Runtime: Python 3.11.9, PyTorch 2.6.0+cu124, CPU only. No CUDA context,
  training, or network access was used.

The replay script validates the frozen corpus manifest and media hashes,
validation sample order and targets, feature-cache shape/finiteness, and strict
checkpoint state loading. Frozen inputs were read only. Replay output is outside
the original run directory.

## Result

| Check | Result |
|---|---:|
| Frozen validation rows serialized | 512 / 512 |
| Selected option IDs matching saved predictions | 512 / 512 |
| Argmax changes | 0 |
| Maximum option probability absolute difference | `4.112720489501953e-6` |
| Declared CPU replay tolerance | `5e-6` |
| Status | PASS, within tolerance |

The replay scores one example at a time while the saved predictions were
produced in batches. CPU floating-point kernels therefore do not produce
bit-identical probabilities across these execution shapes. The observed delta
is below the fixed tolerance; all selected IDs match. This result does not
recompute or amend Accuracy, NLL, Brier, or ECE, and does not upgrade the
previous 96.09% development result into general image, video, multimodal, or
product evidence.

## Output artifacts

External local output directory:
`C:\CodexLocalArtifacts\pretrained-reuse-r2\miniwob-json-runtime-20261116-attempt-01`.

- `validation-decisions.jsonl` SHA-256:
  `46fe5c09a1ec2b6edaa2e75e94276b0bc85a49ca9bf9e65d1b0c723009c0fa8d`.
- `parity-report.json` records all source hashes, count, tolerance, and status.
- Frozen checkpoint, corpus, feature cache, and previous run outputs were not
  modified.

## Remaining work

The next R2 step is an end-to-end wrapper that accepts an image and option
coordinates, runs the pinned frozen DINOv2 encoder, and then emits this JSON
contract. The current API verifies the readout and output mapping over saved
features only. R2's text/audio/video/sensor and both-input-required gates remain
open; this narrow synthetic screen task does not satisfy them.

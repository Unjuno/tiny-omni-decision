# MiniWoB DINOv2 Decision JSON Runtime Parity (2026-11-16)

## Scope

This change exposes the already selected MiniWoB screen readout through the
repository's `classification_json` contract and adds an end-to-end local image
runtime. It replays the existing frozen development predictions as an
integrity check, not a new quality evaluation, checkpoint selection,
calibration fit, or training run. The old Path A results remain the quality
record. Probabilities use temperature 1.0; no calibration was fitted.

`MiniWoBDinoRuntime` strictly loads the pinned local DINOv2-small files and
readout checkpoint, preprocesses a screenshot with the original whole-screen
224x224 resize, extracts the same global/local patch and candidate-coordinate
features, and returns option-ID JSON. A batch API supports efficient local
inference with a consistent number of options per image.

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

The end-to-end command was:

`python scripts/verify_miniwob_dino_image_runtime.py --run-dir D:\CodexArtifacts\pretrained-reuse-r2\miniwob-dino-path-a-v1-attempt-01 --encoder-dir C:\CodexArtifacts\pretrained-reuse-r1\dinov2-small-audit-01\model --output-dir C:\CodexLocalArtifacts\pretrained-reuse-r2\miniwob-image-runtime-20261116-attempt-01`

The wrapper checked all four pinned encoder-file hashes, loaded DINOv2-small
offline with `trust_remote_code=False`, and verified 22,056,576 frozen encoder
parameters. The saved 76,293-parameter readout loaded with strict state-dict
matching. It processed validation screenshots in batches of eight on CPU in
about 27 seconds, including model and corpus integrity checks.

## Result

| Check | Result |
|---|---:|
| Frozen validation rows serialized | 512 / 512 |
| Selected option IDs matching saved predictions | 512 / 512 |
| Argmax changes | 0 |
| Maximum option probability absolute difference | `4.112720489501953e-6` |
| Declared CPU replay tolerance | `5e-6` |
| Status | PASS, within tolerance |

The end-to-end screenshot path was also run over all 512 validation examples:

| Check | Result |
|---|---:|
| Screens decoded and serialized | 512 / 512 |
| Selected option IDs matching saved predictions | 512 / 512 |
| Selected-ID mismatches | 0 |
| Maximum feature difference from the frozen cache | `6.914138793945312e-5` |
| Maximum option probability difference from saved batched predictions | `2.2470951080322266e-5` |
| Status | PASS_SELECTED_ID_REPLAY |

The replay scores one example at a time while the saved predictions were
produced in batches. CPU floating-point kernels therefore do not produce
bit-identical probabilities across these execution shapes. The observed delta
is small and all selected IDs match. The cached-feature-only comparison uses a
fixed `5e-6` probability tolerance; the end-to-end comparison is reported with
its measured feature/probability deltas and does not claim bitwise equality.
Neither replay recomputes or amends Accuracy, NLL, Brier, or ECE, and neither
upgrades the previous 96.09% development result into general image, video,
multimodal, or product evidence.

## Output artifacts

External local outputs:
`C:\CodexLocalArtifacts\pretrained-reuse-r2\miniwob-json-runtime-20261116-attempt-01`
and
`C:\CodexLocalArtifacts\pretrained-reuse-r2\miniwob-image-runtime-20261116-attempt-01`.

- `validation-decisions.jsonl` SHA-256:
  `46fe5c09a1ec2b6edaa2e75e94276b0bc85a49ca9bf9e65d1b0c723009c0fa8d`.
- `parity-report.json` records all source hashes, count, tolerance, and status.
- End-to-end decisions SHA-256:
  `574c046b78217389badee9b7a7220f685d86e99679157d9ec41063e779c1671e`.
- End-to-end verification script SHA-256:
  `b80e37e06e4f977b1640de1b46071f88540fd68ef52f8cbf37f77835ea49d4da`.
- End-to-end report records model/checkpoint/corpus/cache hashes and measured
  feature/probability differences.
- Frozen checkpoint, corpus, feature cache, and previous run outputs were not
  modified.

## Remaining work

The next R2 step is to admit and validate suitable text/audio/video/sensor tasks
and demonstrate both-input-required fusion. This MiniWoB result is a single
synthetic image-screen task and does not satisfy those multimodal gates.

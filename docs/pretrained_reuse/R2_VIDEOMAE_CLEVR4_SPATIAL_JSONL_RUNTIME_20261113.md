# VideoMAE Clevr-4 spatial-grid JSONL runtime — 2026-11-13

## Result

The new task-scoped `image_videomae_spatial` route loaded the pinned VideoMAE
V2 Small checkpoint and the fixed 4×4 Clevr-4 readout, then returned one typed
10-option probability record through the real Path A JSONL dispatcher on CPU.
The probabilities were finite and summed to `0.9999999395`. The route reported
22,153,286 encoder parameters plus 789,962 readout parameters, or 22,943,248
for this image candidate. This is an executable plumbing result, not image
quality evidence: the input was a generated synthetic image, and the predicted
`purple` label has no semantic significance.

The first request used `white`, which is not in the frozen Clevr-4 color
taxonomy, and was correctly rejected before inference. That failed attempt is
preserved separately; no output was overwritten. The second request used the
exact frozen color taxonomy and passed.

## Fixed model and execution

- Runtime integration commit, reproduced after commit:
  `1f66c77d3a37a6f7a93f881e82f059e9a2c3b45f`.
  The initial successful smoke used the same working tree before this commit;
  the post-commit reproduction is attempt 03 below.
- Encoder: OpenGVLab VideoMAE V2 ViT-Small K710, source SHA-256
  `d132c97369f25207e380e78c59294fd00b93d2f6aa0fb5c4ded19eff26a5a3cb`,
  checkpoint SHA-256
  `24fb71687fa3671b8387cadfbcbab0f72af695692e93cf1ecc82caa888626172`.
- Readout: fixed best spatial-grid checkpoint from the seed-17 run, SHA-256
  `10e5f02524f7ae7dd6576950cb896c61fdcb71b270bacc923b82441f85cfc5bf`.
- Runtime: Python 3.11, PyTorch 2.6.0+cpu, timm 1.0.15 from the existing
  isolated R1 dependency tree. The previously used Path A runtime venv lacks
  timm; it was left unchanged. No package was installed or upgraded.
- Device: CPU. The CUDA process belonging to the separate Orbitune job
  (PID 34072) remained live; this run did not use or interrupt it.
- Input preprocessing: generated 224×224 RGB PNG; runtime applied Lanczos
  resize, `/255`, mean/std `0.5`, and static-frame repetition to 16 frames.
- Post-commit attempt 03 `worker_inference_seconds`: 0.9981; first worker
  startup: 6.0085 seconds; total one-request CLI wall time: 10.3017 seconds.
  Earlier passing attempt 02 measured 1.9249 / 8.2807 / 13.5345 seconds under
  the same CPU environment. These single-run values are smoke measurements,
  not a latency benchmark.
- PyTorch reported no CUDA runtime (`torch.version.cuda=null`); the request
  returned `[1, 6144]` features and a typed JSON choice with ten options.
- Upstream timm deprecation and model-registry replacement warnings were
  emitted during dynamic source import. No load, forward, or serialization
  error occurred.

Command shape:

```powershell
python -m tiny_omni_decision.cli path-a-jsonl `
  --assembly <existing-verified-model/assembly> `
  --videoprism-python <existing-python-executable> `
  --videoprism-source <existing-source-directory> `
  --videomae-model-dir <pinned-videomae-v2-small> `
  --videomae-spatial-readout <best-readout.pt> `
  --videomae-dependency-dir <pinned-timm-1.0.15-dependency-root> `
  --worker-log-dir <new-empty-directory> --device cpu
```

The VideoPrism arguments are required by the shared dispatcher constructor but
were not used by this image-only request.

## Preserved outputs

- Failed option-validation attempt:
  `D:\CodexArtifacts\pretrained-reuse-r2\videomae-spatial-runtime-cpu-smoke-20261113-attempt-01`
  (request, rejection response, input image, stderr and worker log retained).
- Passing attempt:
  `D:\CodexArtifacts\pretrained-reuse-r2\videomae-spatial-runtime-cpu-smoke-20261113-attempt-02`.
- Post-commit reproduction:
  `D:\CodexArtifacts\pretrained-reuse-r2\videomae-spatial-runtime-cpu-smoke-20261113-attempt-03`.
- Attempt 03 `result.json`, JSONL response, and request SHA-256:
  `2b5a5562f5764b53e34aa8aa71283741c9f6849775070f16900b349db1fc6918`,
  `54265e8ff53091104d5520f3f05a60905f2ccdc1900c4a16286b49e06ade0655`, and
  `8be6db472e7920c8c3e68bf261c25216d3ccd31d202663dd1cbd6def77f00a61`.
- Attempt 03 synthetic input SHA-256:
  `41f49898e4318ad76df437ac0a3ad7fd0f8fb076b38136d5f25b2cc0804a9421`.

## Verification and limits

- `ruff check` on the changed runtime, CLI, dispatcher, worker, and focused
  tests: pass.
- `pytest -q tests/test_videomae_spatial.py
  tests/test_videomae_spatial_runtime.py tests/test_path_a_jsonl_runtime.py`:
  18 passed. The new CPU unit tests cover caller option order, frozen weights,
  media-hash rejection before feature extraction, and unknown taxonomy options.
- The broader dispatcher compatibility slice, including
  `tests/test_runtime_dispatch.py`, `tests/test_videomae_runtime.py`, and the
  runtime dependency contract, passes: 29 tests.
- `git diff --check`: pass.
- This does not evaluate any Clevr-4 validation/reporting split, prove general
  image reasoning, or alter the fixed model-selection result. It does not add
  the spatial readout to the shared deployment manifest or establish the
  combined Omni package size.
- The previous Clevr-4 spatial selector metrics remain development-only and
  must not be retuned on those selector IDs. Text/audio/video/sensor and joint
  multimodal gates remain open.

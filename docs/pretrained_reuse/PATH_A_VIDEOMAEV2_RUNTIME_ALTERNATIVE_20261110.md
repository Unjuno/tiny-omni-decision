# Path A Torch-only video runtime alternative — 2026-11-10

## Decision

The Path A JSONL smoke cannot start the pinned VideoPrism Image/Video worker in this Windows environment. Windows Code Integrity events 3077 and 3033 at 2026-11-10 05:11 (local time) identify an explicit enterprise policy denial while Python tried to load:

`C:\\CodexArtifacts\\pretrained-reuse-r1\\videoprism-base-r1\\venv\\Lib\\site-packages\\tensorflow\\python\\_pywrap_sanitizers.pyd`

The DLL is unsigned (133,120 bytes; SHA-256 `2b398abc933d8fd6b35d7b52b6330d45a4070696ef9988a87e5ea623cb95bcb0`). Code Integrity reported policy ID `0283ac0f-fff1-49ae-ada1-8a933130cad6`. The failure occurs before media decode or VideoPrism model load. No policy, DLL, package, or environment was modified, and no alternate path was used to evade the block.

As a separate Path A candidate, the already audited OpenGVLab VideoMAE v2 Small checkpoint passed a CPU strict-load and serialization/reload parity smoke using the existing Torch/timm environment. This is a technical runtime result only. It has not been integrated into the Path A dispatcher, and no Decision quality was measured.

## Pinned model and runtime evidence

- Model: OpenGVLab VideoMAE v2 ViT-Small Kinetics-710 distilled.
- Owner model card declares Apache-2.0; retain attribution and license notices. This audit does not promote the model to a product artifact.
- Existing pinned source SHA-256: `d132c97369f25207e380e78c59294fd00b93d2f6aa0fb5c4ded19eff26a5a3cb`.
- Checkpoint: 44,334,609 bytes; SHA-256 `24fb71687fa3671b8387cadfbcbab0f72af695692e93cf1ecc82caa888626172`.
- Strict-loaded parameter count: 22,153,286.
- Environment: PyTorch 2.6.0+cu124, timm 1.0.15, PyAV 18.1.0, NumPy 2.4.6. No package upgrade.
- Device: CPU only; `CUDA_VISIBLE_DEVICES` was empty and the smoke did not initialize CUDA.
- Inputs: deterministic synthetic RGB. The Image path repeated one static image for 16 frames; the Video path used 16 distinct synthetic frames. Preprocessing was `/255`, mean/std 0.5, shape `[1,3,16,224,224]`.
- Both routes returned finite `[1,710]` logits and `[1,384]` features. A serialized state dict (88,667,886 bytes) was loaded into a fresh strict model; logits and features were bitwise identical on both routes.
- Runtime: 11.22 seconds for load, serialization, reload and four synthetic forwards. This is not a product latency benchmark.

Script SHA-256: `8f7dfbecef94c1fc3ef79eafa23bd73fb2d4530fd8e7c9178b525c962fcd626b`.

Run artifact: `C:\\CodexArtifacts\\pretrained-reuse-r2\\path-a-videomaev2-runtime-smoke-20261110-attempt-01\\result.json`, SHA-256 `1055a85e38d2ae4975f5e4f65b0372a9b62c1e8e8a73e6f0b476a1a053c7399f`.

## Scope and limitations

- No training, dataset access, validation scoring, checkpoint selection, or sealed-audit access occurred.
- This smoke proves that the fixed VideoMAE v2 weights can encode synthetic Image-like and Video-shaped inputs and survive state serialization/reload in the tested local CPU environment.
- Its 710-way Kinetics logits do not yet map arbitrary caller-supplied text options to calibrated Decision probabilities. A task-specific option mapping and clean admitted evaluation set are still required.
- The image route's repeated-frame convention is a plumbing contract, not evidence of useful static-image classification.
- The Path A JSONL dispatcher and original assembly remain unchanged. The VideoPrism route remains policy-blocked on this host.

## Verification

- `python -m ruff check scripts/audit_path_a_videomaev2_roundtrip.py tests/test_path_a_videomaev2_roundtrip.py`: pass.
- `python -m pytest tests/test_path_a_videomaev2_roundtrip.py -q`: 3 passed.
- The original pinned VideoMAEv2 R1 runtime report remains unchanged.
- The existing Path A integration change is isolated on this branch; no legacy worktree, PR, dataset, checkpoint, or cache was overwritten.

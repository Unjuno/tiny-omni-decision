# PE-Core T16 CPU state-dict save/reload parity — 2026-10-22

Status: **PASS for the bounded in-memory state-dict parity check.** This is a
runtime integrity smoke, not a Decision-task quality evaluation.

## Procedure

- Source branch/commit: `codex/roadmap-scalar-audio-update-20261022` at
  `6e711eadc3fea182491c1e7d138ea5760734a2fd`.
- Model: `timm/PE-Core-T-16-384`, pinned revision
  `7fe539ed578ac49a1c2b4f946e4b0747704c825a`.
- Local safetensors checkpoint: 278,219,132 bytes;
  SHA-256 `477b7507c79e3484195f4e389af6d044b5d56f42ce503d6b0eb7e99439b545e5`.
- Published licensing declarations: the pinned Hugging Face model card labels
  the PE-Core checkpoint Apache-2.0, Meta's `LICENSE.PE` is the Apache License
  2.0, and the executed OpenCLIP v3.3.0 source is MIT licensed. These are the
  repositories' published software/checkpoint terms; this check does not make
  a separate claim about the provenance or rights of the pretraining data.
  Sources: [pinned PE-Core model card](https://huggingface.co/timm/PE-Core-T-16-384/tree/7fe539ed578ac49a1c2b4f946e4b0747704c825a),
  [Meta PE license](https://github.com/facebookresearch/perception_models/blob/main/LICENSE.PE),
  [OpenCLIP v3.3.0 license](https://github.com/mlfoundations/open_clip/blob/v3.3.0/LICENSE).
- Runtime: Python 3.11.9, PyTorch 2.6.0+cu124, OpenCLIP 3.3.0,
  safetensors 0.8.0, CPU with four PyTorch threads. `CUDA_VISIBLE_DEVICES=-1`
  was set; CUDA was unavailable and remained uninitialized. Hub access was
  offline.
- Constructed the OpenCLIP architecture without pretrained weights, strict-
  loaded all 316 tensors from the pinned checkpoint, then encoded a fixed
  synthetic `[1, 3, 384, 384]` tensor and two fixed text strings.
- Serialized the loaded `state_dict` with `torch.save` into an in-memory
  `BytesIO`, constructed a second fresh architecture, strict-loaded the
  deserialized state, and repeated both forwards.

## Result

- Runtime parameters: **69,546,113**.
- Strict loads: no missing or unexpected keys, both before and after the
  round-trip.
- Image embedding: `[1, 512]`; before/after SHA-256 identical
  (`c7c575b09bca17b7856e4799491ebe2b8336b0f6865e9390a0964c8ad1fd9b44`);
  max absolute difference **0.0**.
- Text embeddings: `[2, 512]`; before/after SHA-256 identical
  (`b17ad68948418426ca957ae42dfc6e97a24b7c58ed1c319beef9e475a83e3b5f`);
  max absolute difference **0.0**.
- In-memory serialized state dict: 278,305,530 bytes; SHA-256
  `305c96343c238f8329c6a4c6c0e351db3574c5b7c2abf27e2365699b2b021574`.
- Elapsed time: 2.499 seconds including checkpoint hash calculation, two
  model initializations, strict state loading, and four forwards.
- Reproduction script, per-stage progress, and full result are preserved at
  `C:\CodexArtifacts\pretrained-reuse-r1\pe-core-t16-r1-attempt-04-memory-reload\`.
  Script SHA-256:
  `881ed57579cebfb1fcb81ff4aa83ebbb872b18c52b88daaf1560b9f2e382b7fd`.
  Result SHA-256:
  `5097ffda7a44d8e7db9792de6df12856e0ec72308c427e95291f1927bb0926bb`.

## Limits and next gate

This confirms the in-memory `torch.save` / `torch.load` state-dict route for
one synthetic tensor and fixed texts. It does not verify a disk-backed package
loader, the official image preprocessing path, full evaluation integration,
Decision quality, task calibration, or end-to-end save/reload through the Path
A dispatcher. The 278 MB FP32 checkpoint itself exceeds the provisional 200
MB whole-package target; quantization or packaging savings are not established
here. The previous attempt-03 remains preserved as an incomplete attempt; this
successful result does not rewrite its history. No existing model, dataset,
attempt, worktree, or live GPU process was changed.

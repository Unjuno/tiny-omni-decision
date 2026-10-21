# Shared-visual candidate: actual simultaneous residency

## Scope and result

The selected components were simultaneously resident in local CPU workers.
VideoPrism image/video share one JAX parameter tree; PyTorch holds MiniLM, a
Cross-Encoder with a saved replacement applied in memory, Whisper and DyMN.
Existing image/video/audio/sensor/fusion checkpoint tensors are also retained.
No source checkpoint changed. No model training occurred and no audit data were opened.

Conservative combined model-parameter plus retained-head-element count: **195,420,792**.
Remaining to 200,000,000: **4,579,208**. Counts are before quantization.
This verifies capacity feasibility for these components only. It is not proof
that a complete, high-quality Omni runtime fits: several heads are only retained
as tensor mappings, no joint decision was executed, screen quality is unverified,
and future required modules still consume the remaining budget.

## Actual ledger

| Component | Parameters / conservative retained elements |
|---|---:|
| Shared image/video VideoPrism | 114,365,184 |
| minilm | 17,389,824 |
| crossencoder | 22,713,601 |
| whisper | 37,760,640 |
| audio_events | 1,961,239 |
| image checkpoint tensors (including buffers) | 61,480 |
| video checkpoint tensors (including buffers) | 400,128 |
| sensor checkpoint tensors (including buffers) | 2,380 |
| fusion checkpoint tensors (including buffers) | 421,995 |
| audio_readout checkpoint tensors (including buffers) | 344,321 |

Parameter identity is deduplicated only for the same live PyTorch parameter
object, never merely for matching content hashes. Distinct equal model copies
count separately. Shared storage through distinct Parameter wrappers would be
overcounted, not silently subtracted. Whisper tied embeddings are counted once.
Checkpoint head counts intentionally include buffers/unused tensors. In particular
the video checkpoint has 400,128 tensor elements versus 393,984 model parameters.

## Two concrete issues found and corrected

Attempt `omni-resident-budget-01` is retained unchanged and is superseded:
- Its numeric control `best-delta.pt` contains 10,647,169 existing encoder/classifier
  tensor values, not that many additional model parameters. The filename is not
  an architecture contract. Attempt 02 requires all keys/shapes to match the
  Cross-Encoder, applies them in memory, verifies exact replacement and releases
  the extra source dictionary. No numeric-embedding module is added by this
  particular control checkpoint. This does not select its text quality.
- On Windows the venv launcher PID differs from the actual JAX Python PID.
  Attempt 01 sampled the launcher memory and understated RSS. Attempt 02 verifies
  the child-reported PID belongs to its live process tree and checks its command
  line, then samples the actual worker plus launcher. Do not cite attempt 01 RSS.

## Runtime observations

Both workers were live at the snapshot. Summed RSS: 1,808,855,040 bytes
(approximately 1.81 GB decimal). This is a point-in-time process RSS sum, not
peak physical RAM, deployment package bytes or a deduplicated OS memory measure.
Shared DLL pages may be double counted. Each process memory structure is saved.
CPU float32 encoder weights and float64 image readout; original environments
were retained without installation or upgrades. No GPU or cloud compute.

Synthetic finite forwards passed for MiniLM, Cross-Encoder, Whisper, DyMN and
both image/video profiles of one VideoPrism instance. These use zero/constant
inputs, not new quality data. The individual task heads were not wired or
quality-evaluated. Image/video timing may reuse compilation caches and is not
a cold-start or end-to-end performance guarantee.

## Next integration gate

Use this evidence to prioritize a typed worker interface with one owned visual
instance and explicit live-model accounting. It must route real inputs to
existing heads with saved-output parity before any claim of an integrated model.
No removal of text/audio paths merely to meet size; no new coefficient search.
Condition semantics, multimodal fusion, quality targets and quantized package
gates remain open. Do not promote the tested collection as the product.

## Reproduction

Source commit: `bdfd6026b67ec5c35943618681e3161a080880c2`.
Command: `python -m scripts.probe_omni_resident_budget --output C:\CodexArtifacts\pretrained-reuse-r2\omni-resident-budget-02`.
All checkpoint hashes, measured counts, worker PIDs, forward shapes, memory
records and runner hash are in the adjacent JSON. The test creates and releases
only its own child process; no pre-existing process was interrupted.
The same command needs a fresh output directory. Archived attempts are preserved.

## Verification

Ruff passed; 585 CPU tests passed, 1 skipped. Model/dataset/catalog validation
passed. GitHub Actions is checked on the pushed result commit.

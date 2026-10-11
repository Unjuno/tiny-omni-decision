# Path A modular runtime assembly — 2026-10-11

## Scope

This is a local packaging and relocation check for the existing frozen Path A
Audio, CLEVR-4 Image, and synthetic Typed Decisions Text runtimes. It does not
train, select, recalibrate, or change any model. The result is a set of
task-scoped components, not a four-modality runtime, a temporal video decision
runtime, or joint audio-video inference.

## Assembly

- Builder source commit: `5fbe5c2232812441a80c956fcdde33043a59642a`.
- Builder: `scripts/assemble_path_a_modular_runtime.py`.
- Assembler and verifier: `src/tiny_omni_decision/modular_bundle.py`.
- Standalone Text runtime: `src/tiny_omni_decision/text_runtime.py`.
- Assembly: `C:\CodexArtifacts\pretrained-reuse-r2\path-a-modular-runtime-assembly-v3`.
- Package manifest SHA-256: `9d98b4384c57ba7b15ae531b3b209dd13677e64baea005ac64fd35b7a534d7ae`.
- Source Audio manifest SHA-256: `574640d88d6ddbbdb2d12f2466a248ce13c8d9eb3ac542d099b2f08981b690f4`.
- Source Image manifest SHA-256: `958e1f8735c33237342d0533f7529a0514fc4f002ca812057b2eff28ac51012e`.
- Text task manifest SHA-256: `d30a845589f9dee62711d2f82597091d4aaaf27d6cea9d77099d4a2f2c067838`.
- All source bundles were checked before copying; the output passed a full
  file-set, size, hash, path and parameter-accounting verification. Existing
  source and v2 directories remain unchanged.
- The manifest records `170,204,547` unique runtime parameters. The sum of
  per-task loaded parameters is `187,594,371` because Audio and Text each load
  the same MiniLM weights in their separate task runtime. These shared weights
  are stored only once in the assembly and referenced by path and hash.
- Package files excluding the root manifest total `686,056,517` bytes;
  unique-content bytes are `686,052,694`. The full directory is about 686.1 MB.
  The provisional 200,000,000-byte target is **not met**.
- Components: Audio uses Whisper Tiny, MiniLM and its fixed task head; Image
  uses VideoPrism-B and its CLEVR-4 head; Text uses shared MiniLM plus a
  separate synthetic-text readout. There is no component for temporal video.

## Runtime relocation checks

The following end-to-end replays used existing development validation records
as packaging checks, not as new model selection or quality evidence. Each
component was loaded from the assembled v3 path and compared with its existing
runtime. Prediction files and reports are outside Git under
`C:\CodexArtifacts\pretrained-reuse-r2`.

| Component | Validation slice | Backend / time | Resource | Parity |
| --- | ---: | --- | --- | --- |
| Audio | 8 audio observations, 32 decisions | RTX 3080 Laptop; load 0.544 s, inference 0.730 s total (0.0913 s/audio) | Peak CUDA allocation 380,225,024 bytes | 0 prediction mismatches; max logit delta `3.34e-6`, max probability delta `1.19e-7` |
| Image | 8 images, 32 decisions | JAX CPU; load 3.576 s, inference 13.975 s total (1.7469 s/image) | Peak working set 4,252,143,616 bytes | 0 prediction mismatches; max probability delta `2.32e-6` |
| Text | 8 synthetic decisions | CUDA; load 0.844 s, inference 0.263 s total (0.0329 s/decision) | Peak CUDA allocation 102,433,280 bytes | 0 prediction mismatches; max probability delta `7.15e-7` |

Report SHA-256 values:

- Audio: `72b8032b0bccb67e2da6e990b7eaa5702664682124852f69583233833d4f88c0`
- Image: `59fe7dadcbb4c74218cd09bd5d4c3569d3842ea0b30457651a644f9e3768c518`
- Text: `c753da530ee3867e9aa582f6dd536d2454504d321762544ab8ac547f7547f1f8`

The Text replay prediction SHA-256 is
`b6a6739ed115cf673154b4f8427df0923fb4f5da321f8298449b5cd9397e3e07`;
the Image replay prediction SHA-256 is
`b9764c0901099f1269319c3f54525b6fa205fbae61c24c03e94c1c4c5a2bde35`.

## Runtime and quality limits

- Audio and Text used Python 3.11.9, PyTorch 2.6.0+cu124, Transformers 5.6.2,
  and CUDA 12.4 on the local NVIDIA GeForce RTX 3080 Laptop GPU (16 GiB).
- Image used the existing separate Python 3.11.16 environment with JAX 0.6.2,
  TensorFlow 2.18.1 and NumPy 1.26.4 on CPU. The assembly records separate
  runtime profiles and does not vendor environments or provide one dispatcher.
- The task scopes remain narrow: Audio is closed-set Speech Commands with
  four fixed query templates; Image is a synthetic CLEVR-4 taxonomy task; Text
  is synthetic Typed Decisions Synth. The tiny replay slices establish
  relocation parity only, not independent quality estimates or release gates.
- The Text task reuses the exact MiniLM artifact already stored in the Audio
  component; no model weight file is duplicated. Text and Audio remain
  independently loaded runtime components.
- Temporal video inference, joint audio-video task support, a common runtime
  dependency profile, and package-size reduction remain open. No training,
  quantization, sealed-audit loading, cloud use, or model-weight update
  occurred.

## Verification

- `verify_modular_bundle` passed for assembly v3, including component manifests,
  path safety, file hashes and parameter accounting.
- Audio, Image and Text relocation replays each had zero prediction mismatches.
- Local full suite: 220 passed, 1 skipped.
- Ruff: passed; dataset catalog validation: 7 training and 9 evaluation
  sources; `git diff --check`: passed.
- Model weights and media remain under `C:\CodexArtifacts`; no large artifacts
  were added to Git.

## v4 Video feature-input scorer addendum

The v2/v3 records above remain the history of those immutable artifacts. On
2026-10-11, v4 added a CLEVRER Video Decision scorer that consumes frozen,
pooled VideoPrism features extracted from 8 uniformly sampled frames. The
scorer itself accepts those feature vectors and the question/candidate text;
raw-video decode and VideoPrism execution are not included in that component's
runtime. The checkpoint is referenced from the already packaged Image
component, not copied again. The Video runtime reports near-chance performance
on its existing development validation and is recorded for interface and
packaging parity only.

- Package: `C:\CodexArtifacts\pretrained-reuse-r2\path-a-modular-runtime-assembly-v4`.
- Source commit: `b32d2b0c3d3fccfeb7772ca61ec153ce99aa1a6a`.
- Package manifest SHA-256: `79fbca05e6062b66fe6ffb8a831d6c635edffe361c76ea4ce1574f0736ae9519`.
- Components: Audio, CLEVR-4 Image, synthetic Typed Decisions Text, and
  CLEVRER binary-choice Video feature scorer. Audio/Image weights, MiniLM and
  VideoPrism are each stored once; Text and Video reference the shared MiniLM
  file from Audio, and Video references VideoPrism from Image.
- Unique runtime parameters: `170,499,716`; summed per-component accounting is
  `319,644,548` because shared encoder parameters are referenced by more than
  one task. Package files excluding root manifest: `687,254,772` bytes;
  unique-content bytes: `687,247,126`. The provisional 200 MB target is not met.
- The Video task-local scorer loads `17,684,993` parameters (MiniLM plus its
  readout); including the shared VideoPrism encoder, its task uses
  `132,050,177` unique parameters. It is not a raw-video standalone runtime or
  an audio-video task.

### CLEVRER Video scorer replay

The v4 packaged runtime was run against all `3,014` frozen binary validation
choices from `200` CLEVRER scenes/videos, using the matching cached VideoPrism
feature per media hash. Validation IDs and order match the saved predictions:
`209acf5d75fbbfa4b017f0e81604590ed549108d2b8519aaab74de3bd45ef283`. The
VideoPrism checkpoint hash is
`61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29`, the
MiniLM hash is
`cf1e4e2d420c664973037c3c73125d7a8fc69952495093ef8f50596f8943a433`, and the
selected readout hash is
`c15f9184794717e5942ed45a2eec18e93b12ad9a15d4b27f8b3cdf4ff480be77`.

| Metric | Result |
| --- | ---: |
| Prediction mismatches against saved readout | 0 / 3,014 |
| Maximum absolute logit delta | `4.77e-7` |
| Maximum absolute probability delta | `1.79e-7` |
| GPU | NVIDIA GeForce RTX 3080 Laptop GPU, CUDA |
| Runtime load time | 0.912 s |
| Scorer inference for all choices | 1.323 s (0.439 ms/choice) |
| Peak CUDA allocated memory | 202,064,384 bytes |

Inference time starts after pooled VideoPrism features already exist. It does
not include MP4 decode or VideoPrism feature extraction, so it is not an
end-to-end video latency measurement. No metrics were reselected or recomputed;
the upstream binary scorer's held validation macro Accuracy was `0.4990`,
NLL `0.6943`, Brier `0.5012`, and ECE-15 `0.0243`, near binary chance. This
validation had previously informed readout selection, so these are not blind
quality estimates.

Replay report SHA-256:
`2598645b66d1abf77aab5e30944125bef7bf3e04f29a756dfbb1eba9c31a8efb`.
Predictions SHA-256:
`de1b00f04a65bc96dd6c105f8ab76f5f1db07aa3346d760ea7aacaf3d53cbfa2`.

`verify_modular_bundle` passed the complete v4 file/hash/path/accounting check.
The v4 Video replay used the assembled MiniLM and Video readout files. Audio,
Image and Text v3 relocation results remain valid for v4's identical component
manifests; the v4 assembly changed only by adding the Video task and manifest
metadata. Joint AV, raw-video extraction through a packaged/dispatcher path,
and package-size reduction remain open. All model weights and media remain
outside Git.

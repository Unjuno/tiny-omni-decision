# R1 frozen-component CPU recheck — 2026-10-26

## Scope and outcome

Re-ran pinned local checkpoints for three frozen feature components in fresh
processes, CPU-only, using deterministic plumbing inputs. This confirms that
the cached artifacts still load and produce finite outputs on this machine.
It is not a task-quality evaluation, does not admit a natural AV source, and
does not close R2 or any product-quality gate. No training or sealed-audit
access occurred. No model weights, corpora, existing run, or checkpoint were
modified.

Project source: `2538c5a6f3702e8535c339984a43af9b9d293fed`
(`docs/pretrained-reuse-roadmap-20261010`). The local worktree was detached at
that source while the audit ran; this report is on a separate branch.

## Components rechecked

| Component | Pinned source / revision | Parameters | Checkpoint bytes / SHA-256 | CPU result |
| --- | --- | ---: | --- | --- |
| VideoPrism base, 8 frames | `google-deepmind/videoprism@c6eba0ece3dd49a24ffc44582966997a4a59f1d8` | 114,365,184 | 457,475,296 / `61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29` | Finite `[1,2048,768]`; deterministic output SHA matched the prior audit; 8.12 s, peak process working set 3,912 MB |
| Ruri v3-30m | `cl-nagoya/ruri-v3-30m@24899e5de370b56d179604a007c0d727bf144504` | 36,705,536 | 146,828,152 / `e94155e342cbfc33280c345c6e2912905fb5de6e1a3adeb938f395aafd2f8832` | Japanese and English plumbing strings finite; hidden `[2,32,256]`, pooled `[2,256]` |
| EfficientAT DymN04-AS | `fschmid56/EfficientAT@a425fdce92572e602a1d5634799bd9f1f2efa806` | 1,961,239 | 8,086,973 / `49b0e63577021994257b3de2ac5f1686d71554b93c8149f9d2568ad5760b904e` | Strict state-dict load; 440 Hz generated 2 s fixture finite logits `[1,527]`, pooled `[1,384]`; 1.25 s |

The three feature components total 153,031,959 parameters and 612,390,421
checkpoint bytes, excluding tokenizer/preprocessor and any task readout. Thus
the provisional 200M *pre-quantization parameter* budget remains plausible for
this particular component combination, while its raw checkpoint bytes already
exceed the provisional 200 MB package target. These are separate constraints.
The recheck does not establish that this is the final architecture or that
compression will meet the package limit.

The EfficientAT checkpoint has MIT code provenance and a checkpoint-owner
statement permitting commercial use ([upstream issue #44](https://github.com/fschmid56/EfficientAT/issues/44)).
VideoPrism source code is Apache-2.0 at the pinned revision. Ruri v3-30m is
recorded as Apache-2.0 in the existing pinned component audit. These checks
are scoped to the stated artifacts, not a comprehensive legal review.

## Environment and exact outputs

- VideoPrism: Python 3.11.16, JAX 0.6.2, Flax 0.11.1, TensorFlow CPU 2.18.1;
  device `TFRT_CPU_0`.
- Ruri/EfficientAT: Python 3.11.9, PyTorch 2.6.0+cu124, Transformers 5.6.2;
  CPU-only. No GPU was used.
- VideoPrism result: `C:\CodexArtifacts\pretrained-reuse-r1\recheck-20261026-0015\audit-result.json`, SHA-256 `09c2c3d34f7a2042499a8c598f42dba5aadf1ff5ce5762592d2d5fb2a951ee53`.
- VideoPrism driver: SHA-256 `fd364264da0f668bbffe8959bdd353c323e1430ef2616aacc78580c206aeb01`; it was modified from the earlier audit driver to write to a fresh result path.
- Ruri/EfficientAT result: `C:\CodexArtifacts\pretrained-reuse-r1\recheck-20261026-0015\ruri-efficientat-result.json`, SHA-256 `582812a0e017c2d6d95300d5e95ff26d26d508f15aa6461f94cf107907c1e28a`.
- Ruri/EfficientAT driver SHA-256: `479f566189af0dca7fbc9ce00a841221f6d745b10c3697b2bfbea0d7e11c4153`.
- Generated audio fixture SHA-256: `fd878fb5531d04a35fd3641f63bb6df1de628e49cc251ca465dc833a878ac8a9`.
- MiniLM and MOMENT files were not present at the expected cached paths during this recheck; no new claims about those components are made.

## Interpretation

This is an artifact/runtime revalidation only. The English Ruri string is a
plumbing fixture and carries no English quality claim. The generated tone is
not a speech or environmental-audio benchmark. The single VideoPrism smoke
input carries no video accuracy claim. No multimodal decision head was trained
or evaluated here. Existing R1/R2 reports remain the source for task results;
this note only adds fresh runtime and hash evidence.

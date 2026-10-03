# Phase 3 — Bounded high-precision Decision LoRA

## Result boundary

The first reproducible bounded local experiment has completed on the pinned Gemma 4
E2B QAT base. It demonstrates the four real processor/model input paths, finite
LoRA gradients, a learning diagnostic, adapter checkpoint save/reload, and metrics
on the frozen evaluation set. It is deliberately capped at 128 distinct selected
training examples and 16 optimizer steps; it is not a durable teacher-quality run.
No ternary conversion, Recovery LoRA, teacher cache, or base-model merge was done.

## Frozen corpus

- Seed: `17`; option ordering is deterministically shuffled by seed and sample ID.
- Per-source cap: 64 source rows, further constrained by each catalog's row limit.
- Train JSONL: 216 decisions, SHA-256 `6fe0372690fe484244239190bf0f7c0afb318f27744ab4b846ad5b8197a99dca`.
- Evaluation JSONL: 211 decisions, SHA-256 `6066a96fd4acf1021979ec9c40d5fc77db6b2d7a46b1a21a8a3f0b7320de75ad`.
- Corpus manifest SHA-256: `3073b756114904fbec8b8b590c087484aa6769dc510f8f95939509e6657fbfca`.
- Train/eval pair SHA-256: `e0955e590bb41537cb1b6414704ae576fbadf413e18f74cb24b76c11e01a0a8d`.
- Split check: 0 shared source IDs and 0 shared normalized-content fingerprints.
- All selected media has been materialized locally and each image/audio/video member is individually SHA-256 recorded in `data/processed/phase3-frozen/corpus-manifest.json`.
- The Clevr-4 and CLEVRER archives were range-read only for selected members. Their entire multi-GB archive hashes were not recomputed locally; this is recorded in the corpus manifest. The source annotation/question files and catalog/source revisions are hashed and pinned there.

Composition:

| Modality | Train | Evaluation |
|---|---:|---:|
| Text | 124 | 119 |
| Image | 32 | 32 |
| Audio | 16 | 16 |
| Video | 44 | 44 |
| **Total** | **216** | **211** |

| Source | Train | Evaluation | Pinned revision |
|---|---:|---:|---|
| Typed Decisions Synth | 65 | 58 | `5ece89a225b23c4cd5c4bab5735a0819d61dd7d5` |
| Open-Jev redistributable release | 59 | 61 | `538ce45e3e888d3e9147d425509e06ff2d05813c` |
| Clevr-4 | 32 | 32 | `cddc78fb2a8359dc958987b2c750bfdd4bfd2c73` |
| Speech Commands v0.02 | 16 | 16 | `a751309c0fd613e8a5d30d77900f30e8b42bc2da` |
| CLEVRER | 44 | 44 | `98b842082ba4f7c18b6b9e3f39145871782a65ef` |

MMAU and MVBench are excluded from training. OneJev is excluded because its
component-level rights remain review-gated. Open-Jev rows with soft/non-single-winner
targets are excluded by the adapter. The catalogs and underlying dataset manifests
are in `manifests/phase3-training-corpus.yaml`,
`manifests/phase3-evaluation-corpus.yaml`, and `manifests/candidates/`.

## Training setup and artifacts

- Base: `google/gemma-4-E2B-it-qat-q4_0-unquantized`
- Revision: `6befbaca7398925921802abd1f277b495b78b738`
- GPU: NVIDIA GeForce RTX 3080 Laptop GPU, 16,384 MiB.
- Environment: Python 3.11; PyTorch 2.6.0+cu124; torchvision 0.21.0+cu124;
  Transformers 5.6.2; PEFT 0.21.2; Accelerate 1.15.0.
- LoRA: rank 16, alpha 32, dropout 0.05; target module paths resolved from the
  loaded pinned model; base and modality encoders/projector frozen.
- Loss: CE weight 1.0 plus Brier weight 0.2; no teacher KL.
- Seed 17; selected 128 training examples, all 211 evaluation examples; 16 optimizer
  steps; micro-batch 1 and gradient accumulation 2.
- Consumed in the run: audio 6, image 6, Open-Jev text 6, Typed Synth text 7,
  CLEVRER video 7 (32 microbatches total).
- Peak allocated VRAM: 11,097,939,456 bytes (about 10.33 GiB); no OOM.
- Adapter and complete `run-metadata.json`, baseline and LoRA raw prediction JSONL,
  optimizer/checkpoint state, and step evaluations are in the ignored local directory
  `artifacts/phase3-bounded-mixed/`. Model artifacts are not committed.
- Tiny-overfit diagnostic: 8 examples, all four modalities, 8 steps; objective
  decreased from about 3.5634 to 2.4449; adapter reload verified. Its predictions
  are a plumbing diagnostic only.

### Evaluation on the same 211 examples

| Metric | Untouched base | Decision LoRA |
|---|---:|---:|
| Accuracy | 0.4028 | 0.5782 |
| NLL | 2.4746 | 1.2021 |
| Brier | 0.9166 | 0.5866 |
| ECE (15 equal-width top-label bins) | 0.4024 | 0.2047 |
| Mean confidence | 0.8052 | 0.7806 |

| Modality | N | Base accuracy | LoRA accuracy | Base Brier | LoRA Brier | Base ECE | LoRA ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Text | 119 | 0.4706 | 0.6723 | 0.8637 | 0.5111 | 0.4185 | 0.2128 |
| Image | 32 | 0.1563 | 0.3750 | 1.1645 | 0.7984 | 0.5017 | 0.3267 |
| Audio | 16 | 0.6875 | 0.9375 | 0.4541 | 0.1094 | 0.1797 | 0.1010 |
| Video | 44 | 0.2955 | 0.3409 | 1.0476 | 0.8103 | 0.4433 | 0.3070 |

The metrics describe this small sampled dataset and training budget; they should
not be interpreted as benchmark-generalizing quality claims. The full source-level
metrics and raw example outputs are in the local run metadata/prediction files.

## Reproduction

Freeze (requires source media/data access) and train:

```bash
tiny-omni-decision freeze-corpus
tiny-omni-decision train-decision \
  --train-manifest data/processed/phase3-frozen/train.jsonl \
  --eval-manifest data/processed/phase3-frozen/eval.jsonl \
  --config configs/decision/e2b_qat_lora.yaml \
  --output artifacts/phase3-bounded-mixed
```

The run uses deterministic modality/source round-robin selection, with no teacher
cache. PyTorch 2.6 is necessary: Transformers 5.6.2's actual Gemma 4 multimodal
attention masking raises an error with the repository's former 2.5.x pin.

## Remaining gates

- Increase training coverage and steps before treating the LoRA as a durable teacher.
- Expand and independently audit held-out evaluation/benchmark coverage.
- OneJev component-level rights remain unresolved and excluded.
- CI for the proposed branch/PR must pass.
- Ternary architecture/runtime compatibility and compression quality are later-phase
  gates and were intentionally untouched here.

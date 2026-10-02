# Tiny Omni Decision

Tiny Omni Decision is an open project for building **small, calibrated multimodal decision models** for edge and mobile deployment.

The initial target uses Google's Gemma 4 E2B QAT checkpoint as a pretrained multimodal alignment backbone, adapts it to typed decisions over **text, image, audio, and video**, then compresses the decoder toward an extreme ternary representation while preserving decision quality with a separate recovery adapter.

The project is intentionally **not** optimized for long-form generation or long chain-of-thought reasoning. The core product is a fast probability distribution over user-supplied options.

## Initial target

- Base candidate: `google/gemma-4-E2B-it-qat-q4_0-unquantized`
- Role of the base: reuse pretrained multimodal alignment, not maximize generative reasoning
- Decision output: yes/no, choice, and score-style probability distributions
- Quantization target: decoder-focused ternary `{-s, 0, +s}`
- Recovery target: restore decision probabilities and calibration after extreme quantization
- Deployment target: local PC first, then mobile/edge runtimes

The exact upstream revision is deliberately not hard-coded yet. It must be pinned in the model manifest before any durable training run.

## Why this project

Extreme compression is most useful when it changes deployment behavior. A small decision model can stay resident on-device, run alongside other models, and process frequent text/image/audio/video decisions without paying the cost of long autoregressive generation.

## Current status

**Phase 0 design is locked enough to start Phase 1.** The repository scaffold, manifests, configs, CLI validation, and CI are being established.

Before training begins, three hard gates remain:

1. pin the exact upstream model revision;
2. freeze a redistribution-safe dataset subset with per-source licenses;
3. verify the exact ternary runtime path for the selected architecture instead of assuming a format is supported.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -e ".[dev]"
tiny-omni-decision validate-model-manifest manifests/base-model.example.yaml
tiny-omni-decision validate-dataset-manifest manifests/dataset.example.yaml
pytest
```

For model work:

```bash
python -m pip install -e ".[ml]"
```

See [ROADMAP.md](ROADMAP.md) for the execution plan.

## License

Repository code and original project files are licensed under Apache-2.0.

Upstream model weights and datasets keep their own licenses. See [THIRD_PARTY.md](THIRD_PARTY.md).

# ARCHIVED — withdrawn standalone Audio Decision subsection

> **WITHDRAWN, 2026-10-12:** The user canceled the independent <=10M/98% audio-only model. All prospective scope, training, model selection, packaging, runtime and Omni-integration discussion below is **historical and no longer actionable**. The active effort is the <=200M-parameter **full Omni Decision model**, including native audio, visual, text and sensor input, crossmodal fusion, and quantization recovery. Preserve old evidence; do not launch an audio-only project based on the older text.
> [Archived standalone roadmap](../audio_decision/ROADMAP.md); [active fusion experiment plan](R2_MODALITY_NATIVE_FUSION_PROTOCOL_20261012.md).

## Superseded historical note

# Audio Decision — independent 10M / 98% research track

> **Status update (2026-10-12).** The previous R2-A "Omni roadmap subsection" wording is **superseded**. The user explicitly chose a **standalone Audio Decision model**, **not** an Audio encoder required to be integrated into Tiny Omni Decision. The prior version of this file remains available in Git history for research provenance.

**Authoritative standalone goal and execution gates:** [Standalone Audio Decision Roadmap](../audio_decision/ROADMAP.md).

## What the user wants

Develop a small, independent model that takes **spoken instructions/criteria and arbitrary currently supplied options** and directly selects the appropriate option or returns ordered option probabilities. Examples include speaking a changed constraint ("the cheapest", "the nearest", "not smoking", "only open now") and selecting among the same candidate collection. The model must be conditioned on the **actual audio**, question and current candidates; not merely a fixed speech-keyword classifier.

- **Complete pre-quantization model <=10,000,000 parameters**, including acoustic Encoder, candidate/criterion representation, any text embeddings, scoring layers, adapters and original-head exceptions. An 8M acoustic Encoder **plus** a 17M language Encoder does **not** qualify.
- **98% Accuracy is a target** on independently grouped, predeclared representative audio-to-choice tests, not a currently achieved result. Per-task performance, NLL, Brier, ECE, conditions, labels and uncertainty must remain separate. Random answer/majority-class, original VAD or keyword scores do not certify the target.
- Prefer pretrained **~8M audio latent representation** with a minimal conditional head and no broad foundation-model retraining. Candidates: [Pipecat Smart Turn v3](https://github.com/pipecat-ai/smart-turn), a pretrained ~8M Whisper-Encoder-based **turn-end detector**; [EfficientAT](https://github.com/fschmid56/EfficientAT) smaller pretrained event encoders. Neither is proven to encode arbitrary spoken conditions. Verify exact checkpoints, actual hidden states and applicable language coverage, or select a better speech-semantic checkpoint.
- **No required Omni integration**, shared weights, audiovisual fusion, parent-model size, prior Gemma Teacher, or parent release dependency. The independent project can be designed and evaluated even if Omni is unfinished.
- Keep all previous experiments unchanged. Standalone Audio Decision uses separate configs, checkpoints, model manifests, frozen splits, reproducibility reports and export/runtime.

## Why this differs from the earlier audio probes

The 97.85% Speech Commands four-query score was measured using **55.49M** combined Whisper Tiny + MiniLM + head parameters on a **closed ten-word set** and three deterministically derived binary questions. That suggests bounded sound-to-decision tasks can be learned with limited additional supervision but **does not measure or prove** ~8M/10M arbitrary spoken-choice understanding.

The same old keyword-trained model transferred poorly to more general LibriSpeech transcript-selection tasks. The STARSS23 experiment included synchronized audio/video and found the joint readout worse than the strongest audio-only baseline. Both are historical diagnostics, not requirements to integrate this **independent** product.

Source details: [R2 feature probes](R2_PATH_A_PROBES.md), [LibriSpeech transfer](LIBRISPEECH_HARD_NEGATIVE_TRANSFER_20261011.md), [STARSS23 comparison](R2_STARSS23_COMPARISON_20261011.md).

## Recommended immediate experiment

First obtain and inspect real pretrained ~8M speech-model weights and **usable intermediate acoustic/semantic features** (a binary end-of-turn ONNX output alone is insufficient). Build a task-matched same-audio/different-criterion/candidate-change test with held-out speakers and candidates. Train a <=2M independent local option encoder and scorer with the base frozen. Only attempt bounded acoustic adaptation if the frozen feature lacks task-relevant information. The final runtime must load without Omni and count **all** parameters toward 10M.

**Change scope:** Documentation clarification only. This file is now a compatibility pointer and evidence summary; the standalone roadmap is the source of truth. No code, model, live experiment, evaluation set, or parent roadmap is changed by this textual update.

# Shared text-backbone feature smoke

Technical feasibility only, no quality evaluation or training. The existing
22,713,601-parameter Cross-Encoder plus saved encoder-coverage initialization
ran on CPU, explicitly hiding CUDA in its separate process. Active deterministic
training, configs and artifacts were not changed. This initialization is not a
selection of the ongoing numeric experiment.

The same resident backbone serves classification and attended-token mean pooling.
Forty unchanged CLEVR-4 candidate prompts produce [40,384] FP32 features, maximum
15 tokens. Repeated features are exact; classification logits before/after are
exact; parameter object identities and frozen weight SHA-256 are unchanged.
Added parameters: zero. Timed load/extraction/hash check: 1.339 seconds. All source
and feature artifact hashes are in the adjacent JSON.

Tests exercise masked pooling, no duplicate parameter objects, no state changes,
repeat parity and rejection of trainable/training models and malformed masks.
The helper creates no second text tower and requires an already frozen model.

This does not make previous MiniLM-trained heads compatible. Before product use,
freeze a candidate checkpoint and retrain/compare readouts on identical training
and validation conditions. Do not infer semantic quality from matching 384-wide
features. The whole-system 198.3M proposal remains unimplemented and unqualified.

Reproduce on CPU with `python -m scripts.smoke_shared_text_features --output <new-dir>`.
Saved artifact: `C:/CodexArtifacts/pretrained-reuse-r2/shared-text-features-smoke-01`.

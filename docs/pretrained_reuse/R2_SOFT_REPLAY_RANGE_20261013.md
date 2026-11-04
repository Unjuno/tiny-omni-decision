# Fixed numeric-range transfer after soft replay

Inference only on the existing hashed10,240-question five-arm development corpus.
256 underlying states,2048 questions/arm, variable candidate counts preserved.
Parent and fixed final soft-replay weights evaluated sequentially on the local GPU
with identical preprocessing; no truncation, training, calibration or selection.
All paired IDs/options/targets and artifact hashes verified. Base weight hash and
complete encoder/classifier checkpoint scope checked before loading.

| Arm | Parent Accuracy | Soft Accuracy | Parent NLL | Soft NLL |
|---|---|---|---|---|
| control |.8462|.8652|.3631|.3957|
| interpolation |.7100|.7192|.8029|.7831|
| extrapolation |.4185|.4326|3.1548|3.0317|
| wording |.8369|.8511|.3862|.3836|
| combined |.4287|.4424|2.7882|2.6198|

Small descriptive changes do not resolve extrapolation failure. Control Accuracy
rises while NLL worsens. No paired uncertainty was calculated for these contrasts;
no claim of statistically resolved improvement. This is a previously observed
synthetic development diagnostic, not a blind task or universal numeric reasoning.
The parent-to-soft contrast includes context adaptation and replay, not loss alone.
Together with the vocabulary result, close this adaptation comparison: in-template
context robustness does not establish semantic or numeric-range generalization.
No promotion, quantization or extra training budget on this evidence.

Commands: `python -m scripts.evaluate_soft_replay_range --parent` then
`python -m scripts.evaluate_soft_replay_range`. Outputs:
C:/CodexArtifacts/pretrained-reuse-r2/coverage-parent-range-01 and
C:/CodexArtifacts/pretrained-reuse-r2/soft-replay-range-01.
Protocols pin input/checkpoint/runner hashes; adjacent JSON contains complete
Accuracy/NLL/Brier/ECE, elapsed runtime and allocated VRAM. Existing artifacts
remain unchanged; repository runner passes Ruff. Whole Omni objective remains unmet.

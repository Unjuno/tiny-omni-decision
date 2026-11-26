# FLAN fixed condition diagnostic

One frozen evaluation of the existing256 questions/64 states/128 opposite-condition
pairs. Same generator, unchanged state/question/options, same prompt body as the
prior SmolLM instruction reference. FLAN uses the plain encoder prompt (no chat
template) and first decoder-token A/B/C logits at temperature1. Labels are distinct
single tokens; no truncation (maximum113tokens). No training, prompt variants,
calibration, seed search, checkpoint selection or sealed data.

Pinned model and local untied-output loader are documented in
`R1_FLAN_T5_RUNTIME_20261013.md`. All loaded weights remain exactly unchanged.

| Metric | Result |
|---|---:|
| Accuracy | .3046875 |
| NLL | 1.1519668 |
| Brier | .7045587 |
| ECE,15bins | .1590562 |
| Both-correct condition pairs | 0/128 |
| Answer-changing condition pairs | 0/128 |
| Predicted A/B/C counts | 256/0/0 |
| Mean full-vocabulary mass on A/B/C | .9877718 |

The model assigns most vocabulary mass to valid labels but always prefers A.
Lower NLL/Brier than the previous instruction reference does not establish useful
condition following. Reject adoption of this frozen scoring setup. No prompt
refinement on these observed questions; this is not a family-wide capacity bound.
It does not justify adding a77M text model to an already over-budget combination.

CPU FP32,4threads; timed model build and inference4.784794seconds, excluding
checkpoint deserialization.76,961,152parameters; no cloud cost. Predictions,
prompts, input hashes and protocol are preserved at
`C:/CodexArtifacts/pretrained-reuse-r2/flan-fixed-condition-probe-01`.
Input SHA2568737946632cf65dd85b6360e9ed1f7b7b70ed7b59df230a0a7f5c9698233fb74.
Run `python C:/CodexArtifacts/pretrained-reuse-r1/evaluate_flan_fixed_condition.py`
with a fresh output directory for reproduction. The script hash is in protocol.json;
it verifies the pinned loader hash and rejects an existing output directory.
Previous results/checkpoints are untouched. This closes the fixed reference
comparison; the overall multimodal quality objective remains unmet.

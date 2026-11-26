# Instruction-pretrained reference: load and fixed diagnostic

Following the user's autonomous-work authorization, assess instruction-pretrained
SmolLM2 as an alternative to retrieval/NLI scorers. This is a research candidate,
not a selected architecture or a claim of multimodal ability.

## Pinned load audit

HuggingFaceTB/SmolLM2-135M-Instruct revision
12fd25f77366fa6b3b4b768ec3050bf629380bac. Publisher card Apache-2.0, English.
[Publisher card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/blob/12fd25f77366fa6b3b4b768ec3050bf629380bac/README.md).
Weights SHA-2565af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c,
269,060,552bytes BF16 storage. Loaded FP32 count134,515,008 parameters;
input/output embeddings truly share storage and are counted once. Local CPU
load/forward1.072sec, no missing/unexpected/mismatched keys, finite output,
repeat logits exactly equal. Tokenizer and all artifact hashes in adjacent JSON.

That leaves65,484,992 parameters under the200M hard limit for all visual/video,
audio, projection and head components. The current114M visual tower cannot be
retained in that budget. A smaller, pretrained visual path would require its own
rights, temporal and quality audit; no modality is silently removed to fit.

## Fixed Decision diagnostic

Same256 questions,64 states, seed17 as previous condition probe; input hash
identical. Use publisher chat template, unchanged state/question, options A/B/C,
and fixed instruction to return only a label. Verify each label is a distinct
single token; normalize their next-token logits at temperature1. No generated
rationale, training, calibration or prompt search. Source commit5393a86 includes
prompt tests proving no target leakage. All inputs fit(max129tokens).

Execution PASS; conditional decision quality NOT DEMONSTRATED:
Accuracy30.47%, NLL1.61631, Brier0.97412, ECE0.45143. All256 predictions A;
0/128 answer changes and0/128 both-correct opposite-condition pairs. Mean
unconstrained vocabulary probability mass on A/B/C is0.4530; the reported
probabilities are conditional on choosing these labels, not calibrated open
vocabulary confidence. This fixed scorer does not meet the product contract.

RTX3080 Laptop FP32,3.250sec timed load/inference, peak PyTorch allocation
1,373,190,656bytes. Original checkpoint unchanged. No cloud cost or sealed access.

## Consequence

A larger instruction checkpoint alone did not fix the approved diagnostic.
This does not establish a universal capacity ceiling, and the study changes
pretraining/model/readout together. Stop selecting language models from public
benchmarks or repeated wording changes. Any next learning experiment must use
independent grouped training/development data, predeclared relational tasks and
learning curves; preserve this diagnostic as a known development failure.
Do not claim95% from specialized synthetic drills or change the four-modality
objective to accommodate these results. Full-system quality remains unresolved.

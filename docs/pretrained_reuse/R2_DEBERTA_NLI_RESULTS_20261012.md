# Compact DeBERTa fixed NLI diagnostic

ExecutionPASS; numerical conditional ranking NOT demonstrated.
Same256questions/64states/3options and byte-identical rendered premise/hypothesis
pairs as the earlier NLI reference. Native entailment index0 verified; FP32,
temperature1 softmax across candidate entailment logits, no calibration/training.

| Condition | Count | Accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| overall | 256 | 0.304688 | 1.143027 | 0.700077 | 0.204097 |
| price:lowest | 64 | 0.375000 | 1.090865 | 0.661853 | 0.135143 |
| price:highest | 64 | 0.265625 | 1.155950 | 0.710980 | 0.210229 |
| risk:lowest | 64 | 0.312500 | 1.168247 | 0.714562 | 0.261258 |
| risk:highest | 64 | 0.265625 | 1.157047 | 0.712913 | 0.209760 |

Opposite-condition answers changed0/128pairs; both-correct0/128.
All256 predictions choose index0(A), matching78 gold labels. No condition following.
Earlier235M NLI referenceAccuracy32.03%; this70.83M candidate30.47%. Both fail
to establish useful numerical reasoning here; the small difference is not a
statistical ranking or a general architecture-capacity ceiling.
Uniform3-way expected Accuracy33.33%, NLL1.09861 and Brier0.66667 are
descriptive references; this candidate probability score is worse.

Two simple sentence-pair smoke outputs were sensible, but this fixed ranking
diagnostic fails. Loading correctly is not sufficient for task capability.
Do not promote, tune prompts, switch scoring or train on this diagnostic.
Model-card task scores cannot substitute for these local measurements.

No truncation: maximum79tokens. Timed2.506s, peakallocated458183680bytes,
CUDAFP32; cloud charges0, electricity unmeasured. Source9dcfe02.
Revision262ae02f29173eec1c250f90804dc7edc677dcff; weight SHA in protocolJSON.
Actual70,830,722parameters includes all embeddings. With current VideoPrism
andWhisper the sum exceeds200M before readouts; no integration/adoption.
Artifacts `C:/CodexArtifacts/pretrained-reuse-r2/deberta-xsmall-condition-probe-01`; full probabilities/hashes preserved.
No image/audio/video or sealed audit evaluation. Four-modality95% remains unmet.

# Annotation disagreement in the fixed audio-question task

Read the already-frozen development manifests, verifying all three split hashes.
No data, model, labels, majority rule or checkpoint selection changed. Three
explicit binary votes per question are retained, including disagreements.

| Split | Questions | Unanimous | Disagreed | Pairwise vote agreement | Fleiss kappa |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 2312 | 1352 | 960 | .723183 | .435782 |
| calibration | 550 | 310 | 240 | .709091 | .403367 |
| validation | 560 | 331 | 229 | .727381 | .444893 |

For each question, observed pairwise agreement is
`sum(n_label*(n_label-1))/6`. Average across questions; chance agreement uses
the pooled yes/no vote proportions. Kappa is `(observed-chance)/(1-chance)`.
This descriptive calculation does not assume that rater identities are known;
the source provides no persistent identities for these individual annotators.

Reused exact saved question-only probabilities and checked source/question order.
On331 unanimous questions: Accuracy .685801, NLL .583523, Brier .396879,
ECE .061804. On229 disagreed questions: Accuracy .593886, NLL .680255,
Brier .485557, ECE .074581. These are diagnostic partitions of all560 original
questions, not replacement benchmarks or hyperparameter-selection targets.

Pooled vote agreement with the majority is .861592/.854545/.863690 for
train/calibration/validation. **This is not a mathematical model Accuracy ceiling.**
A model might reproduce majority labels more accurately than a randomly selected
individual vote; conversely shared mistakes or ambiguous questions can affect
apparent quality. The model-selected development checkpoint and repeated use of
these questions limit generalization claims. No removal of difficult questions,
confidence-based filtering, new label collection or redefinition of the95% goal.
The fixed full audio comparison remains pending media integrity completion.

Artifact: `C:/CodexArtifacts/pretrained-reuse-r2/clotho-aqa-annotation-audit-01/`.
Exact executed code: `clotho_aqa_annotation_audit.py` in the parent directory.
Adjacent JSON includes counts, hashes and both diagnostic metric partitions.
CPU analysis only; no inference or training. CI for preceding optimization
commitb44fefc succeeded on both checks. Existing media acquisition remains live.

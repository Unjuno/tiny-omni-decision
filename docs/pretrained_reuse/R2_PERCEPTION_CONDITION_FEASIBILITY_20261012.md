# Train-only condition-pair feasibility

The pinned Perception Test train annotations contain7,392 three-option questions.
A content-only audit finds498 same-video/same-normalized-option-set question
pairs, all with different question strings;319 have different answer **text**.
Option positions are remapped before comparing labels. This avoids treating
permuted option indices as changed decisions.

Examples include beginning/end letter order and first/second written letters.
These suggest temporal condition tests, but also require readable visual text.
Lexical option equality does not prove equivalent referents, opposite criteria,
audio relevance or adequate coverage by eight frames. No pair is yet admitted
as a product task, and no model output or validation annotation was inspected.
Do not select only the319 changed-target pairs for a headline Accuracy: unchanged
targets and irrelevant-condition controls are also needed to detect blind flips.

Source archive SHA256
`9eaa6aad3fabdf4cde4955f333cfddf3b9e6839c5abcb271afa410870f640d78`.
Command `python -m scripts.audit_perception_condition_pairs --output <fresh.json>`.
Actual output: `C:/CodexArtifacts/pretrained-reuse-r2/perception-condition-pairs-audit-01.json`.
It saves counts, first12 candidate examples, source/script hashes and scope flags.
No media acquisition, training, inference, sealed audit or source changes occurred.

Next admission work: inspect temporal evidence requirements/cuts and classify
candidate-pair semantics on train only, then fix a rule independently of model
performance. Keep upstream scene splits and group both questions together.
This offers a concrete condition-sensitive direction beyond recording matching;
it is not yet a completed quality benchmark or a claimed multimodal capability.

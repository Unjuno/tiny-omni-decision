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

## Template and input inspection

The full498 pairs span358 train videos and29 ordered question-template pairs.
The largest family has99 pairs asking the hidden object's initial/final location
in an occlusion game. Other families include first/second/third shown objects,
written-letter order, counts, and yes/no shape comparisons. They cannot all be
treated as one controlled temporal intervention.

The existing256-recording train acquisition contains67 lexical candidate pairs.
Sixteen acquired videos contain an occlusion-game question;13 have a nonnegative
upstream reveal-cut index. These counts are feasibility information, not sample
selection by model performance. The extended reproducible audit saves all29
template counts in `perception-condition-template-audit-01.json` under the R2 root.

A contact sheet of the existing eight selected frames for the first acquired
occlusion example, `video_3334` / `window-016.npz`, was inspected. It shows three
similar cups and manipulation, but does not clearly establish the initial hidden
object identity or every intervening movement. Therefore this inspection cannot
certify that the eight-frame representation makes the label recoverable. The
sheet is `perception-condition-contact-01.png` in the R2 artifact root. This is
one train-example input audit, not a model score or proof that all clips fail.

Do not fit this family on the existing pooled vectors yet. Admission needs an
explicit temporal evidence audit against the underlying pre-cut sequence and a
prospectively fixed input policy. Preserve the current eight-frame AV experiment
as-is; any different temporal policy would be a separate experiment.

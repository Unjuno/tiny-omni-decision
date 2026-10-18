# Video parent-question diagnostic

Saved predictions from `shared-text-video-01` were aggregated without loading
models or changing checkpoints. All eight condition files have identical IDs,
scene identities, parent questions, task types and targets; hashes are recorded
in [the result JSON](R2_VIDEO_PARENT_DIAGNOSTIC_20261012.json).

There are 3,014 binary choices belonging to 868 parent questions and 200 scenes.
Exact parent accuracy means every **recorded** choice belonging to that question
is correct. This does not prove completeness against the upstream question file,
and no independence assumption is made to multiply binary probabilities into a
parent confidence. Choice accuracy below is micro-weighted, unlike the previous
question-type macro score.

| Condition | Micro choice Accuracy | Parent exact Accuracy |
|---|---:|---:|
| MiniLM + video attention | 51.79% | 14.63% |
| MiniLM + shuffled video | 52.39% | 14.63% |
| MiniLM without video | 51.92% | 13.82% |
| Shared text + video attention | 50.07% | 8.76% |
| Shared text + shuffled video | 50.60% | 7.95% |
| Shared text without video | 50.20% | 7.95% |

Always marking every choice wrong/correct gives 7.83%/7.95% parent exact accuracy.
These simple controls are not upper bounds. Parent accuracy is equally weighted
across questions, so these numbers must not replace the prior macro calibration
metrics. Per-type breakdowns are retained in JSON. No uncertainty significance
claim is made from these point estimates.

The video pipeline remains far from useful conditional reasoning. The MiniLM
model's parent score is unchanged by mismatched video, further cautioning against
interpreting its 14.63% as visual grounding. This is development-data diagnosis,
not a new blind benchmark, capacity ceiling or reason to simplify the target.
Do not continue the same head for more epochs on this evidence. The next useful
intervention must establish visual/temporal information usable by a question
(e.g. object-state grounding) before repeating causal-question training.

Reproduce with `python -m scripts.video_parent_diagnostic`; the script refuses
to overwrite the report. Five CPU tests cover parent aggregation, duplicate
IDs, cross-scene parent conflicts, invalid probabilities and mismatched argmax.
No GPU work, training, sealed audit or model promotion occurred.

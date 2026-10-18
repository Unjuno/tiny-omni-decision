# Question-prior control: grouped uncertainty diagnostic

Reused saved560 validation probabilities and exact saved source/question order.
No inference, fitting, threshold search, checkpoint change or test access.
Compared question-only selected model against the train-frequency constant.
Resampled81 normalized uploader groups with replacement, seed17,10,000 draws;
all questions in each sampled group retained. Compute question-weighted metrics
within each draw, then2.5/97.5 percentile endpoints.

| Difference (question model minus constant) | Point | Cluster interval |
| --- | ---: | ---: |
| Accuracy | +.082143 | [.018908, .141158] |
| NLL | -.061326 | [-.089588, -.032207] |

This supports retaining the question-only control rather than claiming that
any above-prior joint result demonstrates audio use. The checkpoint was selected
on these validation labels; intervals are descriptive development diagnostics,
not blind generalization or selection-adjusted confidence. Questions and clips
within an uploader were not treated as independent bootstrap units.

Local result: `C:/CodexArtifacts/pretrained-reuse-r2/clotho-aqa-question-bootstrap-01/`.
Exact executed code: `clotho_aqa_question_bootstrap.py` in the parent directory.
Full-corpus audio acquisition remains live in its existing session; no restart.
Repository-wide Ruff passed;780 CPU tests passed,1skipped. Both GitHub checks
for4f0ad52 succeeded. These checks cover software, not the unmet multimodal goal.

# Ordinal decision mapping audit

Read-only audit of frozen train/validation and selected Cross-Encoder predictions.
Train 4,943 and validation 545 ordinal questions have list-valued criteria,
integer targets in range, and unique option strings. Every validation prediction
retains the exact source criteria order and source target. This rules out a
source-to-prediction index offset or criteria reorder for these saved outputs;
it does not verify the truth of upstream synthetic labels.

Validation exact matches: 257/545 (47.16%). Absolute index distances:
0:257, 1:183, 2:67, 3:34, 4:4. Of 288 errors, 183 are adjacent and 105
span at least two levels. Mean absolute index error is0.7982; mean normalized
index error is0.2186. These are descriptive: semantic spacing between levels
varies, so neither measure replaces Accuracy, NLL, Brier or ECE.

No relabeling, filtering, revised training objective, new checkpoint or calibration
fit was performed. An ordinal-aware loss is not justified solely by these counts:
remaining causes may include computation, semantics, source ambiguity or missing
state details. A task-specific, independently specified factual diagnostic is
needed before treating this as an architecture/capacity problem.

Reproduction:

```powershell
python -m scripts.audit_text_score_mapping --root C:/CodexArtifacts/pretrained-reuse-r2 --output <new-report.json>
```

Full confusion counts by option count and SHA-256 inputs are in the adjacent JSON.
This is observed development validation, not sealed evaluation. CPU only.

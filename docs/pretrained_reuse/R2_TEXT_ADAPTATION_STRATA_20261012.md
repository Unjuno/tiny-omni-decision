# Saved-prediction adaptation diagnosis

No new inference, model fitting, threshold fitting or sealed evaluation.
Source: completed three-epoch adaptation and fixed zero-shot L6 predictions.
IDs, ordering, options and targets match on all 2,540 questions. Probability
finiteness/normalization and unique IDs were checked. Input hashes are in JSON.

| Question group | Count | Zero-shot Accuracy | Adapted Accuracy | Adapted NLL |
|---|---:|---:|---:|---:|
| Choice | 1008 | 43.75% | 56.65% | 1.07774 |
| Binary noul | 987 | 55.42% | 65.65% | 0.61596 |
| Ordinal score | 545 | 33.03% | 47.16% | 1.26178 |
| Exact instruction seen in train | 394 | 44.16% | 58.12% | 1.01708 |
| Exact instruction unseen in train | 2146 | 46.32% | 58.11% | 0.92323 |

NLL and Brier improve in every listed group. Improvements are not limited to
binary questions or exact repeated instructions. However, exact wording novelty
does not establish semantic/task novelty. Synthetic task families and option
counts differ between groups, so comparing group levels is not causal evidence.

Paired bootstrap resamples 732 independent state IDs with replacement, retaining
all their questions, then calculates question-weighted differences. Seed 17,
10,000 replicates, percentile 95% intervals, adapted minus zero-shot:
Accuracy +12.13 points [10.04,14.21]; NLL -0.26090 [-0.29459,-0.22734];
Brier -0.11544 [-0.13133,-0.09957]. These are posthoc descriptive intervals on
observed validation. They do not correct for best-checkpoint selection or imply
new blind-evaluation confidence. Global ECE remains in the main run report.

Ordinal decisions remain weakest. The result supports task-specific adaptation
of this small text model, but cannot distinguish arithmetic, semantic reasoning,
label ambiguity or state truncation as the remaining cause. Do not enlarge the
training budget or change the architecture solely from this stratification.
Next evidence should independently verify question/state sensitivity and label
semantics, retaining the frozen candidate and existing comparison outputs.

Reproduce with:

```powershell
python -m scripts.audit_text_adaptation_predictions --root C:/CodexArtifacts/pretrained-reuse-r2 --output <new-report.json>
```

No legacy artifacts changed. The audit uses CPU and saved development predictions.

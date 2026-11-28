# Physion++ recorded-factor-only baseline — 2026-11-28

## Purpose and scope

This is a train-only diagnostic of how much the recorded scenario and filename
factors predict the existing Physion++ train labels. It is not a video model
score, a leakage test, an independent validation result, or evidence about the
generalization ceiling. It used the pinned 652-row train manifest only; no
validation/sealed rows, media, model, GPU, fitting beyond the fixed baseline,
or checkpoint selection were involved.

The reproducible implementation is
`scripts/audit_physionpp_factor_only_baseline.py`; CPU regression coverage is in
`tests/test_audit_physionpp_factor_only_baseline.py`.

## Frozen protocol

- Input: `C:\CodexArtifacts\tqpp\fullclip-v1\train.jsonl`
- Manifest SHA-256: `0de1bbbb1cc54000936c1771ea54e375c214fedda36734536f31a44c4f64932b`
- Rows: 652; source asset groups: 492; labels: 339 No / 313 Yes.
- Five deterministic folds: `int(SHA256(source_asset_group_id)[:4])/modulo 5`.
  Every source asset group stays in one fold.
- Comparator: training-fold prior with add-one smoothing.
- Factor baseline: categorical naive Bayes with add-one smoothing, using
  scenario and factors recorded in the source record ID; alpha was fixed at 1.
- Metrics are pooled out-of-fold. This is a single fixed grouped split policy;
  it is not an external or scene-family OOD evaluation. Scenarios and factor
  values can recur across folds.

## Results

| Pooled OOF baseline | Accuracy | Balanced accuracy | NLL | Brier | Probability ECE-10 |
|---|---:|---:|---:|---:|---:|
| Train-fold prior | 0.5199 | 0.5000 | 0.6951 | 0.2510 | 0.0003 |
| Recorded factors only | 0.5982 | 0.5978 | 0.6849 | 0.2416 | 0.0873 |

OOF predictions and machine-readable report are preserved outside Git at
`C:\CodexArtifacts\pretrained-reuse-r2-local\physionpp-factor-only-20261128-attempt-01`.
`audit-report.json` records per-fold counts/metrics, protocol, and hashes;
`oof-predictions.jsonl` has SHA-256
`d85a7a1bd9f222b83b429e53c7560813fd28bd42d88dd9be06c542c02f31c9ce`.
The report records script SHA-256
`c59dbfe54d89babbfaf0502d1e19e0d993db61f4d4d101cc110fe578766c295c`.

## Interpretation and next data work

Recorded factors carry nontrivial predictive signal in this small train corpus.
This does not establish that a video model used those factors, that the dataset
is mislabeled, or that the factors are invalid causal cues. It does show that
overall label balance and correct publisher metadata mapping are insufficient
to establish visual observability or nuisance balance.

Keep model capacity fixed while completing the data review: inspect the
preselected exact-factor matched No/Yes contact sheets for whether the outcome
is visible in the sampled video, and separately review simulator/rendering
regularities. Treat that as manual triage, not a blind evaluation. If labels
are not consistently visually observable, revise the task definition or
construct a rights-cleared, independently labeled dataset before another
architecture sweep. If observability is sound, use source/scenario-held-out
evaluation and report subgroup counts before model changes.

BERT-family work is already broad enough to avoid another generic sweep: the
repository records several task-specific text baselines and a BERT-Mini
capacity check. Those experiments do not diagnose video data quality. Keep
BERT as a text-path reference and do not present it as a fix for video
shortcuts or as evidence for multimodal 95–98% quality.

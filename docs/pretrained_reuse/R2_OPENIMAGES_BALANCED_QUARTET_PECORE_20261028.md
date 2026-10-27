# PE-Core on the existing balanced train quartets

The already inspected 15 four-image train quartets were evaluated once with
the pinned PE-Core T-16-384 checkpoint and the same fixed option wording as
the prior TinyCLIP run. Image IDs, targets and options were matched to all 60
TinyCLIP prediction rows; page receipts, original candidate inputs, image
hashes and the PE-Core weight hash were checked. This is a nonblind exploratory
comparison, not new validation or product selection.

| Condition | Accuracy | NLL | Brier | ECE (15 bins) |
|---|---:|---:|---:|---:|
| PE-Core aligned image | 0.2500 | 2.4087 | 1.0548 | 0.4261 |
| PE-Core neutral image | 0.2500 | 1.4551 | 0.7842 | 0.1342 |
| PE-Core deranged image | 0.2167 | 3.2714 | 1.1578 | 0.4711 |
| TinyCLIP aligned image, prior result | 0.4500 | 1.4162 | 0.6942 | 0.2401 |

Aligned PE-Core confidence averaged 0.6761. Both neutral-image accuracy and
the option-only benchmark are 25% because each quartet has the same ordered
options with each answer correct once. Neutral images are defined by each
model's own preprocessing. This small cohort covers only seven of 11 classes,
is already observed, and does not establish independent photographers, sites,
or absence of pretraining overlap.

Post-hoc paired PE-Core minus TinyCLIP accuracy was -20.0 percentage points;
10,000 seed-17 quartet bootstrap resamples yielded a percentile 95% interval
[-35.0, -3.33] points. The 15 quartets are not established independent source
groups; this interval is descriptive of the resampling assumption, not a blind
population estimate. The two models differ in architecture, preprocessing and
parameter count. No fitting, calibration, prompt search or audit evaluation ran.
PE-Core remains license `REVIEW`; its 69,546,113 parameters and 278,219,132-byte
weights do not by themselves meet the provisional 200 MB package target.

CPU inference used four threads, PyTorch and OpenCLIP 3.3.0, no GPU; elapsed
time including model load and scoring was 17.55 seconds. We used existing
`evaluate_openimages_balanced_quartets.prepare`, the pinned
`evaluate_openimages_pecore_zero_shot.score_conditions`, and fixed metrics from
`evaluate_openimages_tinyclip_zero_shot`. The orchestration was an inline local
Python invocation, retained in the conversation tool record rather than a
standalone repository script. The output is
`C:/CodexArtifacts/pretrained-reuse-r2/openimages-balanced-quartet-pecore-20261028-attempt-01/`.
SHA-256 `summary.json`:
`310b3385cf58863c9ee9fd2d58ca5340e49b66d56ba36ad9b7d714c1cb52c4dd`;
`predictions.jsonl`:
`ee041a7d4bf7bc0436c9d2569433ddbb246a418219c4604791fee4cdc84b5228`.
PE-Core weights SHA-256:
`477b7507c79e3484195f4e389af6d044b5d56f42ce503d6b0eb7e99439b545e5`.

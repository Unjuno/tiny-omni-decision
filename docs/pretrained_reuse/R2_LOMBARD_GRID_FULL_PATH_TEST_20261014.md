# Mechanical full-path fixture: pass, no real quality evidence

An isolated local process tested the full fixed24-epoch/batch128/seed17
three-arm trainer and report orchestration on **artificial one-hot arrays**,
not Lombard Grid features. Sizes4080/398/889 match the production shapes.
Train encodes the target while calibration/validation deliberately shift the
one-hot category by1; this predeclared contradictory fixture exercises early
best selection without stopping the24-epoch budget. It is not noise augmentation,
a dataset variant or a scientific hyperparameter trial.

All three arms completed24 epochs/768 optimizer updates, while retaining best
epoch1 and a distinct final24 checkpoint. Total consumption97,920 assets and
587,520 slot decisions per arm; selected checkpoint consumption4,080/24,480,
with4,080 unique assets and93,840 repeated asset consumptions. These numbers
verify accounting logic on test data; they do not describe the pending real run.

Same-backend best/final replay, every optimizer log, seed17 sampler, train-only
normalizer, calibration-only temperatures, saved full-vocabulary predictions and
six grouped-bootstrap comparisons passed. Weight-preserving matched/mismatched/
missing controls passed. Fixture end-to-end29.567s; peak allocated GPU memory
127,351,808bytes(maximum over arms). This tests the cached-feature head, excluding
real media or encoder costs. No cloud compute, no real validation quality evaluated.

Exact runner`C:/CodexArtifacts/pretrained-reuse-r2/lombard_full_path_fixture.py`;
artifacts under`.../lombard-full-path-test-fixture-01/`, source commit
`f7e8c6a79ef0a2fe81967745e495bb5ce898223f`. Adjacent JSON embeds only mechanical
counts/timings/hash evidence, **not artificial accuracy as native performance**.

The historical evaluator used a shared native-report status/scope string even
though its provenance carried`test_fixture_only=true`. Preserve those original
files and attach`FIXTURE-SCOPE-ADDENDUM.json` explicitly correcting the scope.
The evaluator now emits`PASS_MECHANICAL_FIXTURE_REPORT` and artificial/no-quality
scope for such fixtures; a regression case was observed RED then GREEN.

The real frozen-video extraction remains a separate active process. Its complete
source/hash/reload gates are mandatory before the real three-arm study. The
24-epoch native learning, measured audiovisual benefit and full Omni quality
objective are still incomplete. Do not promote these fixture weights or results.

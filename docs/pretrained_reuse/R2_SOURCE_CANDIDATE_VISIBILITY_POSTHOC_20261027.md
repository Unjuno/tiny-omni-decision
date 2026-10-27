# Source-candidate visibility stratification: post-hoc diagnostic

The [existing nonblind transfer](R2_SOURCE_CANDIDATE_TRANSFER_RECONCILIATION_20261027.md)
was further analyzed from four saved prediction files, hash-checked against the
original receipts. No model, new inference, training, selection, or sealed audit
ran. This post-hoc subgroup analysis cannot establish product quality and its
intervals should not be treated as confirmatory tests.

| Existing condition | Visible source: correct / 59 | None visible: correct / 79 |
|---|---:|---:|
| Joint audio + video | 18 (30.51%) | 54 (68.35%) |
| Joint with audio zeroed | 4 (6.78%) | 72 (91.14%) |
| Joint with mismatched audio | 13 (22.03%) | 44 (55.70%) |
| No-media option prior | 0 (0%) | 79 (100%) |

Joint minus zero-audio accuracy was +23.73 percentage points on the visible
subset (95% paired video-group percentile interval +11.86 to +35.59) but
−22.78 points on the none-visible subset (−32.91 to −12.66). Joint minus
mismatched-audio accuracy on visible examples was +8.47 points, with interval
−3.39 to +22.03. This shows a substantial observed tradeoff in this cohort,
not proof of causal audio reasoning: zero input is an out-of-distribution
intervention, mismatched audio can change several semantics simultaneously,
and neither is a separately trained unimodal model. The sample was label
selected and previously inspected; multiple post-hoc comparisons were made.
Unmeasured clock alignment, oracle object candidates and source labels also
limit interpretation. Do not retune on these results or treat them as a blind
quality gate. A fresh, independently grouped task with matched unimodal models
is still required.

The comparison checks exact record IDs, video IDs and targets across conditions
before pairing. It uses 10,000 within-stratum video-group bootstrap draws,
seed 17; all 138 examples happen to have distinct video IDs. The full result is
`C:/CodexArtifacts/pretrained-reuse-r2/perception-source-candidate-visibility-posthoc-20261027-attempt-01/result.json`,
SHA-256 `f2c03fc7ccd893d138ff197da8af38b9ebd1a8579b3a6dde2e671b60ac4756b6`.
Executed script SHA-256 is
`bff8bef17d21c4ae70eba73be3f80c57417fc24ede2d912be8c3a1d3ae4f9476`.
Two targeted tests and Ruff passed; prior artifacts and predictions remain
unchanged.

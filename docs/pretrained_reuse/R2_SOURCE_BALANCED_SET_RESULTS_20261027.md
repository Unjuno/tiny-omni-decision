# Balanced candidate objective: fixed comparison

The [fixed protocol](R2_SOURCE_BALANCED_SET_PROTOCOL_20261027.md) ran the
same three video-group folds, seed, seven epochs, modes, architecture and
threshold as the ordinary set loss, changing only candidate loss aggregation.
All nine CPU fits completed. These are already observed train-source folds with
oracle tracks and potentially shared contributors, not a blind evaluation or
product test. No thresholds, seeds or checkpoints were selected from results.

| Joint input | Ordinary BCE | Balanced candidate BCE |
|---|---:|---:|
| Exact set, all 2,593 | 23.99% | 18.63% |
| Exact set, none visible (692) | 52.89% | 53.18% |
| Exact set, one target (799) | 23.53% | 4.26% |
| Exact set, multiple targets (1,102) | 6.17% | 7.35% |
| Candidate NLL per record | 0.2283 | 0.3664 |
| Candidate Brier per record | 0.0682 | 0.1212 |
| Candidate micro ECE | 0.0152 | 0.1524 |
| None NLL | 0.4378 | 0.4316 |

For multi-target examples, selected true candidates rose from 1,007 to 1,899
out of 2,451, while missed true candidates fell from 1,444 to 552. False
candidate selections increased from 695 to 3,033. The class-balanced
objective therefore increased candidate recall with a large false-positive
and calibration tradeoff under the fixed threshold. The multi-target exact
increase of 1.18 points has a video-group bootstrap interval -0.96 to +3.43
points; the overall exact decrease was 5.36 points, interval -6.84 to -3.94
(10,000 draws, 320 videos, seed 17). It is not evidence that class weighting
cannot work with other calibrated decision rules, but this fixed candidate is
not selected. Binary candidate calibration is marginal, not a normalized set
probability; micro ECE weights rows with more candidates more heavily.

Balanced audio/video/joint exact-set accuracy was 7.79% / 16.81% / 18.63%;
the audio-only candidate logits are tied and do not identify visual sources.
The joint-minus-video difference was +1.81 points, interval +0.73 to +2.92,
but this is nonblind, shared-source development data and does not establish
that synchronized audio/video correspondence is learned. No sealed audit was
opened. No old files were overwritten.

CPU training took 44.17 seconds and did not initialize CUDA. Nine checkpoint
hashes were verified and strictly loaded; prediction hashes, record identities
and targets were checked for pairing. Prediction replay from weights was not
performed. Run receipt:
`C:/CodexArtifacts/pretrained-reuse-r2/perception-source-balanced-set-cv-20261027-attempt-01/run.json`,
SHA-256 `ee04092410d024692b8123f953bc65c772fffb8f930db03f5a72e633d216267c`.
Summary SHA-256 `33bbe0995f17f856937e1e1846952128526214c70ff8ea607974021d19121620`;
paired comparison SHA-256 `6dec046ab35125acbac61d5c24348c857d75178c3c74d31ec1b2a20ae6422b92`.

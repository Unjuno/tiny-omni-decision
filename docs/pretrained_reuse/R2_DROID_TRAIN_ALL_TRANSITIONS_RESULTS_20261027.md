# Action-change coverage within fixed DROID train recordings

The [fixed train-only procedure](R2_DROID_TRAIN_ALL_TRANSITIONS_PROTOCOL_20261027.md)
re-read nine pinned local TFRecord shards, verified their hashes, and inspected
commands only for the same 275 previously selected train recordings. For each
recording it counted adjacent command labels at target index `t >= 8`, using
the existing dominant-axis rule and `1e-4` native-command threshold. A total of
153 nonselected records were read as opaque records to identify their source,
but their numeric contents and labels were not extracted. No images, model,
validation or audit labels were decoded.

| Train-recording coverage | Count |
|---|---:|
| Recordings | 275 |
| Adjacent pairs with target index >= 8 | 80,998 |
| Same command label | 73,556 |
| Changed command label | **7,442** |
| Recordings with at least one change | **274 / 275** |
| Recordings with at least five changes | 266 / 275 |
| Changes per recording, median / maximum | 18 / 335 |

Copying the previous label would score 90.81% over these train-recording
timesteps, but that deterministic figure is not a held-out metric. The earlier
15 changes came from one fixed target per recording, not from a shortage of
change events over the complete train recordings. Timesteps from the same
recording are correlated and must not be treated as 80,998 independent groups.

| New label on a change | Changed pairs | Recordings containing such a change |
|---|---:|---:|
| stop | 880 | 258 |
| x- | 955 | 238 |
| x+ | 1,398 | 265 |
| y- | 965 | 226 |
| y+ | 953 | 216 |
| z- | 1,154 | 253 |
| z+ | 1,137 | 264 |

These changes use a discretized dominant-axis label: small command fluctuations
or axis ties can create transitions, and their physical significance and time
intervals are not established. The result supports designing a recording-group
split with multiple past-only windows, explicit changed/stable reporting and
controlled sampling; it does not prove independent operator/site coverage,
causal sensor benefit or multimodal quality. A new evaluation procedure must be
fixed before using non-train groups and retain natural-prevalence metrics.

The aggregate-only result is
`C:/CodexArtifacts/pretrained-reuse-r2/droid-train-all-transitions-20261027-attempt-01/summary.json`,
SHA-256 `bcf23e7d2d92d908cd6181a38e9c3b96ea8566c8f4b1238d9a545f3e1adb18b3`.
It includes the full transition matrix, source/protocol/code hashes and
per-recording count distribution. No raw command rows or new training files
were saved. CPU scan and integrity checks took 18.51 seconds.

# RoboPro current-state/action shortcut audit — 2026-11-23

## Outcome

A fixed, no-fit heuristic using only the current robot joint state ranked the
recorded target command first or tied for first on **183.5 of 192 TRAIN
queries (95.5729% fractional credit)**. There were seven top ties. Across the
16 exact-prompt groups, fractional accuracy ranged from 79.17% to 100%.

This is a **post-hoc training-set shortcut diagnostic**, not a validation
estimate and not evidence of policy correctness. It shows that the current
four-choice demonstrator-command task can be solved largely by matching the
current joint state to same-time candidate joint commands. The logged command
is not independently labeled as uniquely correct, safe, or optimal. This
construction therefore cannot support a claim that video is necessary or
useful for the measured answer. Keep this task `REVIEW_NOT_ADMITTED` for R2-M
fusion-quality claims; do not scale the fitted video/state readouts on this
same task.

## Scope and integrity

- Source: Hoshipu/RoboPro `90ec789bf4018eb9c0f75da9f69aab5c185f0fd0`;
  publisher metadata declares CC BY 4.0.
- Input sample: the frozen 16-prompt / 4-episode-per-prompt probe sample,
  manifest SHA-256
  `eeab697bc4189ba2b4442ed53def960f248cf146b2cc947c0633905a41340686`.
- Frozen prompt-split manifest SHA-256:
  `b90196fd2963a5eb90f6e41339f71129c6306fbfd7b45fd80326a5ebdd4c60c1`.
- Download receipt SHA-256:
  `3c5da58198c80478526cde6513c29cd58954416cf19244a715bfeeec2c275ce3`.
- The script read the 64 selected **TRAIN Parquet** files only and checked
  every file's receipt SHA-256, byte length, row count, and `episode_index`.
  It also checked finite 14-D state/action rows and matching publisher channel
  names. Before reading data, it pinned the exact sample manifest, prompt
  split, download receipt, episode metadata, and source `info.json` hashes; it
  also rejected any train/development prompt or episode overlap in the frozen
  sample. The development Parquets and videos were not opened. The dataset's
  metadata does not state numeric units; the score is a channelwise rank rule
  and assumes each same-named state/command dimension shares a monotone
  encoding.
- No model was loaded, fitted, or evaluated. No GPU or cloud compute was used.

The audit was motivated by the published state/action channel names and the
existing small readout-run artifact. Its numeric computation below used TRAIN
rows only; the development predictions did not enter the algorithm or metric.
This distinction does not make the heuristic prospective or independently
validated.

## Frozen heuristic

For each of the 16 TRAIN prompts, use its four sampled episodes and the fixed
progress points 25%, 50%, and 75%. At each query, compare that episode's
14-D `observation.state` with the four same-prompt episodes' 14-D recorded
`action` at the corresponding progress point. Row selection uses Python's
`round((length - 1) * progress)`, matching the frozen RoboPro sample audit.

The two gripper channels (indices 6 and 13) are omitted. For each of the 12
named joint channels, rank the four candidates by absolute state-command
difference using one-based average ranks for exact ties. Sum the ranks by
candidate; the lowest sum wins. A query where the target is tied among *k*
minimum-score options receives `1/k` credit if the target is one of them,
otherwise zero. This fixed channelwise-rank construction avoids arbitrary
cross-channel unit weighting; it does not establish that a retrieved command
is behaviorally valid.

## Result and interpretation

| Measure | TRAIN result |
|---|---:|
| Exact-prompt groups | 16 |
| Episodes | 64 |
| Queries (3 per episode) | 192 |
| Fractional correct | 183.5 / 192 |
| Fractional accuracy | 0.955729 |
| Queries with tied top score | 7 |
| Per-prompt fractional accuracy range | 0.791667–1.000000 |

The high figure is expected to be vulnerable to target construction: the
current joint state and demonstrator's target joint command share named
channels at nearly the same time, and robot commands are locally persistent.
It is a warning about the **candidate target**, not a quality result for
RoboPro, a learned model, or multimodal perception.

An already completed external readout run on this same 64-TRAIN / 64-development
sample is retained as context, not repeated here. Its report records 192
development queries across 16 held-out exact-prompt groups. State-only reached
55.73% Accuracy / 0.8461 NLL / 0.4838 Brier / 0.0750 ECE-15; text+state+video
with a temporal GRU reached 51.56% / 0.9089 / 0.5364 / 0.0698. The paired
10,000-resample prompt-group interval for joint-GRU minus state-only Accuracy
was −9.90 to +2.08 percentage points. This small, exposed development result
does not show video benefit and must not be tuned against. The run's report and
predictions are external artifacts; the exact readout-training script hash was
not captured in that run report.

Together, the train-only shortcut and the weak development comparison give no
reason to spend more model capacity on this action-retrieval target. A future
RoboPro task would need a separately justified target (for example,
demonstrator-vs-counterfactual validity with independently grounded labels),
and auditable scene/task grouping, before any new fitting. Do not infer
valid/invalid alternatives from this record's single demonstrator action.

## Reproduction

```powershell
python scripts/audit_robopro_state_action_shortcut.py `
  --sample-manifest C:/CodexArtifacts/pretrained-reuse-r2-local/robopro-action-probe-v1-attempt-02/sample-manifest.json `
  --split-manifest manifests/pretrained_reuse/robopro-prompt-development-split-20261123.json `
  --data-root C:/CodexArtifacts/pretrained-reuse-r2-local/robopro-action-probe-v1-attempt-02 `
  --metadata-dir C:/CodexArtifacts/pretrained-reuse-r2-local/robopro-metadata-screen-20261123-attempt-01 `
  --output C:/CodexArtifacts/pretrained-reuse-r2-local/robopro-state-action-shortcut-train-20261123-attempt-04/shortcut-audit.json
```

The authoritative external JSON report is SHA-256
`9b91031c586c3dd537bb6414b71db081031c9bfe9d967d5ff485ec8691185c56`.
The audit script at this revision is SHA-256
`a263e856d36922185a81af887937b2e426fb16a2e35246f0b26f2519c3346646`.
An initial scratch execution used half-up rounding and produced a different
score; it is retained outside Git as superseded. Attempt 02 reproduced the
canonical Python `round` result before the final script added an explicit
split-manifest membership/hash gate. Attempt 03 reproduced the same score with
that gate. Attempt 04 is authoritative: it additionally pins every frozen
input artifact hash and explicitly verifies disjoint sampled train/development
identities. It reproduces the same train-only score; attempt 03 remains
preserved externally.

## Existing readout artifact provenance

The referenced external `readout-run-01/run-report.json` SHA-256 is
`e8c98766ddb47d945151c07304cdc7eb3ecc516c95b4e207d4ced95b40831836`. It
records a CPU-only fit, seed 17, frozen VideoPrism-B revision
`be719a406d563b66f0ac969e7c94bab8e997c81a` (114,365,184 parameters) and
MiniLM revision `4ca70771034acceecb2e72475f72050fcdde4ddc` (17,389,824
parameters). Its metric selection was on the exposed project-created
development split, not on an official test or sealed audit. This report neither
recreates that fit nor promotes the source/model.

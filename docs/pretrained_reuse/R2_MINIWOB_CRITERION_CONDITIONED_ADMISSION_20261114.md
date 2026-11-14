# MiniWoB criterion-conditioned Path A admission screen — 2026-11-14

## Decision

**Do not train a Decision head or open the sealed audit for this candidate.**
On all 1,024 frozen validation questions (256 independent screen groups), a
transparent rule that reads the requested CSS color from the question and
matches it to the median RGB at each supplied action center achieved 100%
question accuracy and 100% all-four-criteria-per-screen accuracy. This task is
therefore too directly solvable by pixel equality to measure useful reuse of
pretrained world representations. The result is a task-admission diagnostic,
not a neural-model score or a product-quality claim.

## Frozen task and source

- Task: MiniWoB++ `click-color-v1`, rewritten as four criterion-conditioned
  questions per screen: `Select the <color> button.`
- Each screen group stays intact. The original instruction and target swatch
  are masked; the model-visible action choices expose only opaque IDs and
  normalized centers. Collector-only records retain source labels for integrity
  checking.
- Source: `Farama-Foundation/miniwob-plusplus` commit
  `33c3b4ddef8c6eb67c57a29663d844b1eda7e614`, MIT license SHA-256
  `24c56d96c8a2e4e20c20867b14d177b801027b603c45622218b34c841ad4972a`.
- Config SHA-256: `6b6f914783e04ef7c09b6be32c9d917e0f945527c8e01e0f39f2b7bd03fca36f`.
- Effective config SHA-256: `2474dea1f57c670a4ea30a9425247deeb03426227c54d7cf6fbf79dc6dab1f6f`.
- Corpus code commit: `9af13e4e` (on the isolated R0-refresh branch).

## Corpus integrity

The accepted corpus is attempt 02 under
`D:/CodexArtifacts/pretrained-reuse-r2/miniwob-criterion-conditioned-20261114-attempt-02/`.
It was frozen in 170.841 seconds on CPU; no model was loaded and the GPU was
not used.

| Split | Unique screen groups/assets | Question rows | JSONL SHA-256 |
|---|---:|---:|---|
| Train | 512 | 2,048 | `9817910ee11cf7ad90eb9d46a1609f614843c5c5f8758471e62b91e35054fdf7` |
| Validation | 256 | 1,024 | `82ac8db6c20f2d3b39eebfa7567181be8d696e6b3206cf73001645f345ee9dfc` |
| Sealed audit | 256 | 1,024 | `31e0772377f7f34058b606590750a1f5cbd43faeadeaafcad3e62832442e45a8` |

The sealed row hash is copied from the creation manifest; the sealed examples
were not read, scored, or used for selection. Independent checks on train and
validation confirmed file and decoded-pixel hashes, 160x210 dimensions, the
64-row mask, four unique criteria per screen, and zero cross-split seed or
masked-pixel overlap. The source audit recorded 1,001 train seeds to accept 512
groups and 548 validation seeds to accept 256 groups; rejected seeds were due
to layout eligibility or duplicate source pixels. Thirteen color words were
represented in both train and validation.

## Failed first write attempt

Attempt 01 generated the same fixed-window screen candidates but failed when
writing collector receipts because the `corpus/audit` parent directory had not
been created. Its `attempt-failure.json` marks it `INCOMPLETE_DO_NOT_USE`.
Those files remain preserved and were not used. The collector was corrected to
create the audit directory before retrying. Attempt 02 used a distinct output
directory and completed all integrity checks.

## Validation-only rule diagnostic

The diagnostic parsed the named color from the fixed question text, sampled a
3x3 pixel patch at each supplied action center, and selected the option with
the smallest squared RGB distance to that CSS named color. It used only the
model-visible criterion, image, and option positions. Result:

- 1,024 / 1,024 question rows correct (100.0%).
- 256 / 256 screen groups had all four criterion decisions correct (100.0%).
- No training, temperature fitting, checkpoint selection, or sealed evaluation
  was performed.

The exact screen-color comparison is intentionally a weak task for testing
representation reuse. Keep this corpus and result only as a negative admission
record; do not spend additional model training or tuning on it. Prefer a
source/task where the correct decision depends on nontrivial semantic or
temporal evidence and where a simple input-only rule does not saturate
validation.

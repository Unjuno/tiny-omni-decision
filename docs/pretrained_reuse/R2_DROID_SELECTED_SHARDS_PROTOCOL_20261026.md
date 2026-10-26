# Full DROID selected-shard protocol — 2026-10-26

## Frozen source selection

This is a small, reproducible development sample from the official DROID
training split. It is **not** the source's official validation/test split, a
sealed audit, or evidence of general-purpose Decision quality.

The official `dataset_info.json` at GCS generation `1710518471693866` describes
version `1.4.0`, 92,233 train episodes, and 2,048 shards. Before reading
selected episodes, 12 shard IDs were chosen as
`sorted(random.Random(17).sample(range(2048), 12))`. Their IDs and expected
episode counts are fixed in
[`droid-selected-shards-seed17-20261026.json`](../../manifests/pretrained_reuse/droid-selected-shards-seed17-20261026.json).
The chosen IDs are 110, 451, 715, 1019, 1033, 1139, 1186, 1242, 1497, 1573,
1696, and 1718, totaling 539 expected episodes. Metadata lists 10,724,824,602
bytes across the 12 shards and three pinned release metadata objects. Each
object is pinned by GCS generation, exact size, and MD5.

Dataset-info SHA-256 is
`8cbd8f70428943515a67e2a571267c4de2eb71f33957c0269d8fcd5608951e4f`, the
feature-schema SHA-256 is
`4c556e2e38e179e41089e07cf54273b1540d67d3c78a420898370329813c9a25`, and the
CC BY 4.0 notice is generation `1710518471643551` with SHA-256
`7e7170e3cebf88a9f60c7b8421418323c09304da1af4d5e90f4da1dc1c8a2661`.
The DROID paper states that the full data release is CC BY 4.0. Preserve its
attribution as specified in the [source screen](R2_DROID100_SOURCE_TASK_SCREEN_20261026.md).

## Split policy, before model work

First audit the selected records' source-file and recording-folder identities
against one another and the already-screened DROID-100 debug subset. The debug
subset is not to be mixed into this experiment. Exclude any selected full
record whose recording-folder or source-file identity overlaps the debug
subset. Use exact normalized instruction overlap only to keep duplicate text
content in one partition; report near-synonymous tasks separately.

If at least 400 independent recording groups remain, reserve at least 200
groups for a development validation partition using this deterministic rule:

1. Union records that share either recording-folder identity or normalized
   instruction text.
2. Sort the resulting components by
   `SHA256("droid-r2-seed17:" + sorted(component recording-folder IDs))`.
3. Add whole components to validation until it contains at least 200 unique
   recording groups; use all remaining components for training.
4. Fail closed if validation has fewer than 200 groups or training has fewer
   than 200. Do not weaken the group rule or inspect model metrics to change the
   allocation.

This is a development train/validation split from a source whose official
split is `train`. It is not a final blind audit. Keep any later final audit in
unused assets outside these selected shards. The source sample is only 539
episodes; if group/text collisions leave fewer than the predeclared minimum,
stop before model fitting and report the exact shortfall.

## Acquisition and split audit results

The fixed 15-object acquisition completed successfully in 324.84 seconds.
Receipt SHA-256:
`2bfb6322e9c0c4649f875d8ad37fb3e990f786324fe2fdbd9cd6b2872af50667`.
The metadata-only group audit JSON SHA-256 is
`541595a78e565613b9e1929a2be0fa73dc8f78bb863a71b32a76e81af9cfb7ab`; the
development split SHA-256 is
`d8d2e1ef353d2ea6388dd7e7574623e3b4d6518474afed14f042fc1838df7e43`.
All receipts and split files are outside Git under
`C:\CodexArtifacts\pretrained-reuse-r2\droid-selected-shards-seed17-20261026-attempt-01\`.

| Audit | Result |
| --- | ---: |
| Downloaded objects and bytes | 15 objects; 10,724,824,602 bytes |
| Episodes / unique source files / unique recording groups | 539 / 539 / 539 |
| Per-shard episode counts matching pinned TFDS metadata | 12/12 |
| Success / failure folder labels | 444 / 95 |
| Unique normalized instruction strings / repeats | 669 / 45 |
| Source-file overlap with DROID-100 debug subset | 0 |
| Recording-group overlap with DROID-100 debug subset | 0 |
| Normalized instruction overlap with DROID-100 debug subset | 6 |
| Debug-overlap episodes excluded | 0 |
| Development train / validation episodes and groups | 339 / 200 |
| Train/validation source, group, normalized-instruction overlap | 0 / 0 / 0 |
| Sealed audit from this sample | None |

The 6 matching instruction strings did not cause data rows to be imported from
the debug subset. Within the selected full-data sample, every repeated
normalized instruction was assigned wholly to one side by the union-group
split rule. These results establish deterministic development partitioning,
not task quality or external generalization.

## Temporal-field and language-coverage audit

A second pass used the same pinned bytes and same split rule to count serialized
step fields without decoding JPEG pixels or fitting a model. The enriched audit
JSON is external at
`C:\CodexArtifacts\pretrained-reuse-r2\droid-selected-shards-seed17-20261026-attempt-01\group-audit-03.json`
with SHA-256
`652b9973bb74c39b005d9bd367f9c880957b0271ac0a6e0ccb060cb79470a87f`.
Regenerating the split produces the same SHA-256 as the original:
`d8d2e1ef353d2ea6388dd7e7574623e3b4d6518474afed14f042fc1838df7e43`.

| Coverage check | Result |
| --- | ---: |
| Episode steps (min / median / p90 / max) | 7 / 227 / 534 / 2,238 |
| Mean episode steps | 289.38 |
| Episodes under 8 steps | 1 |
| Instruction present / absent | 302 / 237 |
| Train instructions present / absent | 181 / 158 |
| Validation instructions present / absent | 121 / 79 |
| Episodes where all 3 camera counts equal step count | 539 / 539 |
| Episodes where joint-velocity count equals 7 × steps and gripper count equals steps | 539 / 539 |

Thus text conditioning is available for only 56.0% of this sample. Instruction-
conditioned scores must use the text-present subset and also report text-missing
behavior separately; treating missing instructions as ordinary text examples
would misstate the task. The one 7-step episode is not enough for an 8-step
history and must be explicitly excluded from that windowed analysis. These
checks establish serialized field counts and alignment, not timestamp units,
value semantics, image content, model inputs, or action-prediction quality.
The pinned feature schema places cameras, state, and action in a shared step
sequence, but exposes no explicit per-step timestamp field. Only step-index
alignment can be claimed here; absolute timing, sampling jitter, and clock
offset are unknown and must not be fabricated.

## Candidate Decision task

The task candidate is short-horizon action selection conditioned on the
language instruction, a causal window of synchronized camera frames, and the
robot's proprioceptive state history. The target comes from `action_dict` at
the current step; no future observations or actions may enter the input. Prefer
the explicit `action_dict.joint_velocity` (7 values) and
`action_dict.gripper_position` (1 value) fields rather than assuming the
top-level `action` field selects the desired representation. DROID maintainers
explain that `action` is a default RLDS field while `action_dict` preserves the
available representations ([official schema discussion](https://github.com/droid-dataset/droid/issues/9)).
The task builder must verify tensor semantics, retain typed values, and avoid
unverified unit claims.

Before fitting, freeze and test the sampling rate, camera-view policy, temporal
window, action option construction, hard negatives, target mapping, and choice
shuffle. Then compare identical examples/options with instruction-only,
video-only, state-only, video+state, and all-input models. Require missing,
shuffled-state, stale-state, temporal-offset, and contradictory-input
controls. Fusion is useful only if the actual paired inputs improve on the
best unimodal result and replacing either input causes the expected error
increase. High aggregate accuracy alone does not close R2 M-2.

A reasonable frozen-representation candidate from the existing R1 inventory
is VideoPrism-B (114.37M parameters), English MiniLM-L3 (17.39M), and a small
trained sensor-time LSTM plus gated Decision readout. Whisper Tiny (37.76M)
is the already-audited English speech candidate if a later task includes
audio. These are candidate components, not an adopted architecture or a
quality result. Measure the complete resident parameter count and package
bytes before making any product-size claim.

## Execution boundary

Download only the 12 fixed shards and their release metadata to a new directory
outside the repository. The downloader verifies generation-pinned HTTPS
responses, GCS MD5/size, metadata SHA-256, the 1.4.0 release, and each shard's
episode count. The group auditor reads serialized metadata, instructions, and
protobuf list element counts; it does not decode or display images, extract
model features, or open a sealed audit. No model training or GPU work has
started.
Model work still waits for a tested causal-window/action-option task builder
and matched baseline/control definitions.

The 100-episode debug-source screen is recorded in
[`R2_DROID100_SOURCE_TASK_SCREEN_20261026.md`](R2_DROID100_SOURCE_TASK_SCREEN_20261026.md).
Official sources: [DROID dataset docs](https://droid-dataset.github.io/droid/the-droid-dataset)
and [DROID RSS 2024 paper](https://www.roboticsproceedings.org/rss20/p120.pdf).

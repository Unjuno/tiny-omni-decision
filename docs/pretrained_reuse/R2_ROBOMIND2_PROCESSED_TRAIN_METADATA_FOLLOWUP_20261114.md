# RoboMIND 2.0 processed TRAIN metadata follow-up — 2026-11-14

## Disposition

Keep the pinned derivative `REVIEW_NOT_ADMITTED`; do not acquire more TAR
shards or fit/evaluate a readout yet. This follow-up inspected the pinned
README, shard index, publisher duplicate audit, and only `.json` members in one
TRAIN TAR using byte ranges. It did not access `.mp4`/`.bin` payload bytes,
published-validation records, or model weights.

## Pinned metadata inspected

- Dataset: `OneWorld-AI/RoboMIND2.0-processed` revision
  `99b62f054a4118f1323483b9a78eac697e7d2e85`.
- README: 12,715 UTF-8 bytes at the pinned revision.
- `shards.jsonl`: 90,444 UTF-8 bytes; SHA-256
  `3e9d904d1ef5aae58a48e618a50d7dffa1a559ecc7ae5d70e6ca6a739a31f3ef1`.
  The file lists 608 TAR shards; only rows whose paths are under `/train/`
  were used for the train-size summary below. No validation record contents or
  labels were inspected.

## Schema and task assessment

The README documents one stable episode key and three TAR members per episode:
video (`.mp4`), state/action values (`.bin`), and metadata (`.json`). The JSON
is documented to include task instructions, frame rate/count, row and camera
layouts, source episode identifiers, and other interpretation metadata. It
does not document a stable collection-session, scene, or task-generation group
key. The source says the demonstrations are successful, and `source_aliases`
can retain alternate instructions for one recorded demonstration.

A project-defined next-action task could discretize TRAIN-only demonstrated
actions into typed options and map a held-out demonstrated action to one option.
That would be an imitation-of-recorded-action target, not an optimal-action,
task-success, or causal decision label. The model would need matched
video-only, state-only, and joint controls to show whether both streams help.
The README does not establish that a video/state split can be grouped by an
independent session or task-generation identity, so this task is not yet
admitted.

## TRAIN archive cost bound

The shard index contains 546 TRAIN shards and 121,101 TRAIN episode rows,
totaling 2,334,461,685,760 bytes. TAR shards are the acquisition unit. The two
smallest TRAIN shards total 5,146,368,000 bytes and contain 251 episode rows;
this is only a lower-bound acquisition cost for 200 rows, **not** evidence of
200 independent groups or of task/robot coverage. It excludes extraction,
feature caches, and working space. The full release remains much larger than a
reasonable first local probe. No disk-capacity-dependent action was taken.

## Bounded TRAIN metadata-member extraction

To check the documented per-episode schema without saving or reading media,
the pinned `data/franka/train/00230.tar` was traversed using HTTP byte ranges.
The shard-index row and server `Content-Length` both give 4,222,361,600 bytes;
the CDN honored byte ranges. Only TAR headers and `.json` member bodies were
requested. `.mp4` and `.bin` members were skipped by their TAR-declared sizes;
no range intersected those payloads. The published `validation/` directory was
not queried.

| Check | Result |
|---|---:|
| TRAIN JSON records parsed | 231 / 231 |
| TAR members / headers read | 693 / 694 (includes end marker) |
| HTTP range requests | 925 |
| Bytes returned | 1,154,596 (0.027345% of the TAR) |
| Range scan elapsed time | 350.8 s initially; 287.2 s on repeat |

The 231 JSON records have 231 unique `episode_id` values. The nested `source`
object has `dataset`, `episode`, and `task` fields: respectively 1, 231, and
117 unique values in this shard. The top-level task/description fields have
113 unique instruction values, 113 `instruction_source` values, and 117
`task_name` values. Four records carry `source_aliases`. These are episode,
task, and description identifiers; the card does not establish that
`source.episode` means an independent recording session or scene. No top-level
scene, session, recording, sequence, or group key was present.

The pinned duplicate audit has 48 groups and 11 cross-published-split groups.
Its 72-character episode fingerprints did not directly match the sampled
`source.episode` or `episode_id` strings (zero exact string matches). This
does not prove the sample is duplicate-free: the identifier namespaces are
not documented as equivalent. The audit remains a release-wide integrity
warning; it cannot certify a new TRAIN-only split. Two attempts at hashing a
sorted TAR member-name index produced different digests, so neither digest is
used as a reproducibility claim. A canonical, specified member manifest would
be required before any corpus freeze.

## Gate and next action

Existing evidence still fails the required independent-group and target gates:

1. The pinned README lacks a documented recording/session/scene grouping key.
2. One shard has 231 unique source-episode identifiers, but their relation to
   recording/session/scene identity is not established; these must not be
   counted as 231 independent groups.
3. A prototype-label task can be defined from recorded actions, but it would
   measure demonstration imitation and requires a frozen option/negative
   policy plus source-grouped split.
4. Exact duplicate videos cross the publisher train/validation boundary in 11
   groups; the published validation must remain unused. Any future split must
   be carved from TRAIN only and keep duplicate-connected episodes together.

This bounded JSON screen is complete and does not clear the candidate. Do not
repeat it on another shard unless a concrete way to establish independent
recording/scene groups and a defensible Decision target has first been found.
If a later source-level audit still cannot define a defensible group key and
at least 200 independent development groups after deduplication, close this
candidate without fitting. No model training or evaluation was performed.

## Source

- [Pinned dataset card and schema](https://huggingface.co/datasets/OneWorld-AI/RoboMIND2.0-processed/tree/99b62f054a4118f1323483b9a78eac697e7d2e85)
- [Pinned shard index](https://huggingface.co/datasets/OneWorld-AI/RoboMIND2.0-processed/blob/99b62f054a4118f1323483b9a78eac697e7d2e85/shards.jsonl)
- [Existing duplicate audit](https://huggingface.co/datasets/OneWorld-AI/RoboMIND2.0-processed/blob/99b62f054a4118f1323483b9a78eac697e7d2e85/audit/duplicate_groups.jsonl)

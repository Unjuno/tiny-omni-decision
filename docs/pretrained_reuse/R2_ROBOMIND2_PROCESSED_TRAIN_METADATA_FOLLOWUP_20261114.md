# RoboMIND 2.0 processed TRAIN metadata follow-up — 2026-11-14

## Disposition

Keep the pinned derivative `REVIEW_NOT_ADMITTED`; do not acquire TAR shards or
fit/evaluate a readout yet. This follow-up inspected only the pinned README and
the shard index. It did not fetch episode TAR files, read episode JSON, inspect
published-validation labels or arrays, or access media/model weights.

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

## Gate and next action

Existing evidence still fails the required independent-group and target gates:

1. The pinned README lacks a documented recording/session/scene grouping key.
2. Episode JSON may carry source identifiers, but those per-episode TRAIN
   metadata have not been inspected; their semantics and group cardinality are
   unknown.
3. A prototype-label task can be defined from recorded actions, but it would
   measure demonstration imitation and requires a frozen option/negative
   policy plus source-grouped split.
4. Exact duplicate videos cross the publisher train/validation boundary in 11
   groups; the published validation must remain unused. Any future split must
   be carved from TRAIN only and keep duplicate-connected episodes together.

Before any acquisition, decide whether an explicitly bounded TRAIN-only
metadata extraction is worth the minimum multi-gigabyte TAR transfer. It may
inspect only episode JSON metadata and duplicate/group identifiers; it must
not decode video, state/action arrays, or published validation payloads. If
those metadata do not yield a defensible group key and at least 200 independent
development groups after deduplication, close this candidate without fitting.

## Source

- [Pinned dataset card and schema](https://huggingface.co/datasets/OneWorld-AI/RoboMIND2.0-processed/tree/99b62f054a4118f1323483b9a78eac697e7d2e85)
- [Pinned shard index](https://huggingface.co/datasets/OneWorld-AI/RoboMIND2.0-processed/blob/99b62f054a4118f1323483b9a78eac697e7d2e85/shards.jsonl)
- [Existing duplicate audit](https://huggingface.co/datasets/OneWorld-AI/RoboMIND2.0-processed/blob/99b62f054a4118f1323483b9a78eac697e7d2e85/audit/duplicate_groups.jsonl)

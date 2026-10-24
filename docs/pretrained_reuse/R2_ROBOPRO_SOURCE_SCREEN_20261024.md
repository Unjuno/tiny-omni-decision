# R2 source screen: RoboPro — 2026-10-24

## Decision

**REVIEW; not admitted to Decision-model fitting or quality evaluation.**
RoboPro is a relevant video-plus-proprioception *task-design lead* for temporal
robot manipulation. Its source data is not itself a multiple-choice Decision
benchmark: the primary target is a 14-dimensional continuous action stream,
and the task instruction is also present. No evidence yet shows that a compact
option-probability model can solve a preregistered task using both video and
sensor state, or generalize to independent scenes.

This screen used the public Hugging Face dataset card, README, Croissant file,
and unauthenticated public dataset metadata API. No media or Parquet payloads,
labels, or model outputs were accessed. No dataset download, access request,
training, or GPU work was performed.

## Pinned public metadata

- Dataset: `Hoshipu/RoboPro`.
- Public API revision observed 2026-10-24:
  `90ec789bf4018eb9c0f75da9f69aab5c185f0fd0`.
- API metadata: `gated=false`, `license=cc-by-4.0`, last modified
  `2026-05-01T21:28:52Z`.
- Dataset repository size shown on its public page: 43.5 GB.
- The release describes 80 bimanual manipulation tasks, 3 RGB camera streams
  at 50 Hz, 14-DoF joint state/action streams, and free-form English task
  prompts. The Croissant metadata says the table has 3,728,773 rows/frames;
  the current README says 3,728,445 frames and 15,999 episodes, while the
  latest repository commit says 16,000 episodes and explicitly identifies
  episode 15,999 as a duplicate of episode 12,000. Resolve this metadata drift
  from pinned files before building any manifest; never count the duplicate as
  an independent episode.
- The card/Croissant describe a deterministic clean-vs-cluttered condition
  design. The Croissant explicitly warns that generalization requires held-out
  scenes outside that distribution. It also says collection used one robot in
  a fixed lab with fixed camera/lighting/table geometry.
- The repository has a single `train` split in its public viewer (48k rows).
  These are frame/record rows, not 48k independent tasks or validation cases.
  No official validation or test split was established in this screen.

## Fit to the product objective

The source has useful temporal video and sensor inputs, plus language
instructions, and it concerns real decision/action selection. It may help test
whether a small model uses state history to select among action candidates.
However, converting continuous expert actions to a small candidate-choice
distribution would create a new benchmark target and could introduce arbitrary
candidate-generation/evaluation choices. Behavior-cloning loss on the native
continuous action target would not, by itself, demonstrate calibrated
probabilities over supplied options.

The source is therefore a possible narrow robotics research task, not evidence
for broad text/image/audio/video decision quality. It has no audio modality and
cannot close the required audio-video or audio-video-sensor fusion gate.

## Admission gates

| Gate | Evidence | Status |
|---|---|---|
| Dataset-level reuse terms | Public metadata and card state CC BY 4.0 | Metadata-level pass; retain attribution and verify any third-party/model-specific terms before distribution |
| Revision and provenance | API revision pinned; README/Croissant count discrepancy and known duplicate | Partial; reconcile exact file revisions and episode identities |
| Media/sensor availability | MP4/H.264 and Parquet fields are documented | Metadata only; no payload/decoder preflight |
| Independent split | One train split shown; author notes fixed scenes and calls for held-out scenes | Fail for quality evaluation until a group-disjoint split with scene/variant groups is justified |
| Decision target | Native labels are continuous 14-DoF target joint commands | Not established for option-probability objective |
| Both-input necessity | No matched video-only, sensor-only, shuffled/missing-input controls in this screen | Not established |
| Data uniqueness | Latest commit says episode 15999 duplicates episode 12000 | Must deduplicate by underlying episode/media identity |
| Scope/cost | 43.5 GB full repository; single selected-file cost unknown | Full download not justified; no data accessed |
| Product modality coverage | Video + sensor + text; no audio | Cannot satisfy all-modality objective |

## Next gate

Keep RoboPro at **REVIEW**. The metadata-only episode audit is complete. Before
any media download or model run, establish scene/task/variant group identities
from authoritative metadata, independently resolve the declared duplicate's
media identity, and estimate selected media sizes. Then specify a discrete candidate-action
task whose labels are mechanically derived from the source, whose candidates
are available to the model, and whose evaluation can show that video and
proprioception are both needed. Use episode/scene-group-disjoint development
and held-out evaluation splits; include video-only, sensor-only, missing,
shuffled, stale-state, and prompt-only controls. If a defensible split and
both-input task cannot be built, reject this source for Decision evaluation
without spending GPU time.

## Metadata-only follow-up — 2026-10-24

The pinned public revision was inspected without downloading video, Parquet, or
action payloads. Only `info.json` (3,235 bytes), `episodes.jsonl` (1,443,285
bytes), and `tasks.jsonl` (110,465 bytes) were retrieved into the isolated
local artifact directory
`C:\CodexArtifacts\pretrained-reuse-r2\robopro-metadata-screen-01`.
The reproducible auditor is
[`audit_robopro_metadata.py`](../../scripts/audit_robopro_metadata.py); its
machine-readable summary is
[`R2_ROBOPRO_METADATA_AUDIT_20261024.json`](R2_ROBOPRO_METADATA_AUDIT_20261024.json).

The pinned `info.json` resolves the repository-card count discrepancy for
this revision: it declares 16,000 rows, indices 0–15,999, 3,728,773 frames,
1,622 tasks, 48,000 videos, and only `train: 0:16000`. The current README's
15,999 / 3,728,445 figures are stale relative to this pinned metadata. The
source-declared duplicate episode 15,999 has the same task prompt and length
(328) as episode 12,000; media bytes were not compared. After excluding that
known duplicate, at most 15,999 distinct demonstrations remain.

All 1,622 exact instruction-prompt groups contain multiple episodes (2–79,
median 9), but those prompt groups do not identify scene/configuration groups.
Thus they cannot be used as a scene-independent split. The metadata audit
confirms there is no declared validation or test split. The source remains
REVIEW; no training or quality evaluation is admitted. The raw three metadata
files and their SHA-256 hashes are preserved in the local attempt directory;
the committed JSON summary records their hashes, counts, duplicate claim, and
scope.

## Primary sources

- [RoboPro dataset card and license metadata](https://huggingface.co/datasets/Hoshipu/RoboPro)
- [RoboPro README and schema summary](https://huggingface.co/datasets/Hoshipu/RoboPro/blob/main/README.md)
- [RoboPro Croissant metadata and declared limitations](https://huggingface.co/datasets/Hoshipu/RoboPro/blob/main/croissant.json)
- [Pinned public dataset metadata API](https://huggingface.co/api/datasets/Hoshipu/RoboPro)

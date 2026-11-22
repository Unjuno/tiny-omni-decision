# RoboTwin unified LeRobot source/task screen — 2026-11-22

## Decision

**Status: `REVIEW_NOT_ADMITTED`. Do not download trajectory/video payloads or
train on this source yet.** The metadata suggests useful synchronized video
and robot-state/action streams, but the dataset license is absent on the
candidate repository, the publisher's card says MIT, and the official docs
describe Apache-2.0. The scope of those licenses for the released derived data
and embedded third-party assets is unresolved. The public split is train-only,
and a leakage-resistant Decision task/split has not been defined.

This screen is metadata-only. No Parquet, video, trajectory, model, validation,
or sealed-audit content was opened. No GPU or training run was used.

## Pinned metadata and observations

- Candidate dataset: [`lerobot/robotwin_unified`](https://huggingface.co/datasets/lerobot/robotwin_unified)
- Candidate revision: `1287871839fae2296bc27b88a5457c3e1eba8e1f`
- Publisher source: [`TianxingChen/RoboTwin2.0`](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0)
- Publisher revision: `3dc3b798668feb99ac61cc9086d84cbcc3d79186`
- The candidate repository has no dataset card or `cardData.license` in the
  pinned Hub API response. The pinned publisher card reports `mit`; the
  [official RoboTwin docs](https://robotwin-platform.github.io/doc/) describe
  the LeRobot data release as Apache-2.0. These statements do not establish
  which license covers the candidate's data and embedded assets.
- Pinned `meta/info.json` reports LeRobot v3.0, 27,500 episodes, 6,075,103
  frames, 23,559 task rows, 30 FPS, and only `train: 0:27500`.
- Available modalities are three 480x640 RGB video streams (`cam_high`,
  `cam_left_wrist`, `cam_right_wrist`) and 14-dimensional `observation.state`
  and `action` vectors. No audio stream is listed.
- `meta/tasks.parquet` is listed as 740,974 bytes, but was not fetched or read.
  Consequently task wording, task-to-episode mapping, episode boundaries,
  variation/object/scene identities, and any usable group split remain
  unverified. A train-only frame/episode index is not an independent
  validation or audit split.
- The publisher repository also lists large object/background/embodiment
  archives. Their contents and licensing relationship to the derived LeRobot
  data were not inspected.

## Decision-task feasibility and risks

The native targets are continuous robot actions, not calibrated probabilities
over supplied options. A candidate action-choice task could compare two or
more future action summaries, but its option construction, causal observation
cutoff, label source, and baseline must be fixed before accessing held-out
labels. The question must not reveal the target action. A learned policy or
task-name lookup could otherwise appear multimodal while bypassing visual
evidence.

Before admission, resolve all of the following from permissible metadata or
the publisher:

1. Written license scope for the exact derived dataset revision and included
   assets; do not infer data rights from the code repository's MIT license.
2. Stable episode/task/scene/randomization/object grouping and a deterministic
   grouped train/validation/test design, or an explicit limitation if such
   keys do not exist.
3. A causal supplied-option Decision contract, with an observation window
   ending before the target and a non-leaking task prompt.
4. Whether video adds value beyond state-only and task-text controls on the
   same held-out groups; no quality claim from this metadata screen.
5. A path to the required text/image/audio/video objective. RoboTwin has no
   audio and cannot alone satisfy the product target.

## Reproducibility receipts

Raw API receipts are outside Git at
`C:\CodexArtifacts\pretrained-reuse-r2\robotwin-unified-source-screen-20261122-attempt-01\`.
SHA-256 values:

| Receipt | SHA-256 |
|---|---|
| `lerobot-info.json` | `D9D7880523C2D80FE2CCE95C2E8B6BC198A96ACE7E735C8AED29BA7C1850251C` |
| `lerobot-meta-tree.json` | `7A20967F85E1395186BD16245ED9DDDC293355059F9542DAA8E024CC3ECC57A8` |
| `lerobot-repo-api.json` | `CFCB352666125009F12721E8244F316371BCD676FA9C07FA55CE3FA069CA2214` |
| `lerobot-root-tree.json` | `ABDFBFFECBE18ED94DF9829819E596EE285B52A94AA108514452A9121721C789` |
| `publisher-repo-api.json` | `BA9E4504B9E0A5B8FCB4F8FEA23C58CEE468E75EAC32BD2E6EC9774D4BC67B41` |
| `publisher-root-tree.json` | `ABDFBFFECBE18ED94DF9829819E596EE285B52A94AA108514452A9121721C789` |

The root-tree API receipt is an empty/null response; the repository API
`siblings` listing and pinned `meta/info.json` are the evidence used above.
These receipts establish what was observed, not data-use authorization.

## Next action

Keep the source unadmitted. Ask the publisher or rely on an explicit license
statement for the exact dataset revision, then continue with metadata-only
group/task-contract verification. Do not download the data or use this source
in fitting until those gates pass.

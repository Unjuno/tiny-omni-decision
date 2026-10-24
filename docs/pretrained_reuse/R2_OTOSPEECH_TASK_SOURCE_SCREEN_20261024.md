# R2 source screen: otoSpeech Task-oriented 20h — 2026-10-24

## Decision

**REVIEW; not admitted to fitting or quality evaluation.** This is a promising
new candidate for a bounded audio-plus-screen task, especially
`Spot the Difference`, but it is not a video-input dataset and access is
gated. Public metadata supports the task's structure; it does not establish
that this repository can access and use the files, that independent splits
are adequate, or that a learned Decision model benefits from both modalities.

This screen used the public dataset card, publisher-authored task description,
and the public Hugging Face metadata API only. No gated file, session content,
audio, image, event label, preview session, or model output was accessed. No
access request or contact was sent.

## Pinned public source metadata

- Dataset: `otoearth/otoSpeech-full-duplex-task-oriented-20h`.
- Hugging Face dataset revision reported by its public API on 2026-10-24:
  `383254132dbd8e9465da91f2fe683728c706923f`.
- API metadata: public repository, `gated=auto`; last modified
  `2026-08-31T13:02:19Z`.
- Dataset card states **CC BY 4.0** for the repository contents and reports a
  total release size of 3.88 GB. It says access requires agreeing to share
  Hugging Face account information; access is manually approved.
- The publisher describes 20.005954 hours, 58 two-speaker sessions, seven
  collaborative task types, 11,025 timestamped events (6,115 primary),
  48 kHz per-speaker channels, task stimuli, and reference answers where
  available.
- The dataset card explicitly says the CC BY 4.0 grant does not extend to 82
  third-party YFCC/Flickr images used by `Photo Talk`. Those images are not
  needed for the `Spot the Difference` candidate and must remain excluded
  unless their own rights are independently cleared.
- The card prohibits participant identification or de-anonymization and asks
  users to respect redaction spans. These constraints must be carried into any
  later data handling.

Primary sources:

- [Dataset card and access/license terms](https://huggingface.co/datasets/otoearth/otoSpeech-full-duplex-task-oriented-20h)
- [Publisher's task walkthrough and task design](https://huggingface.co/blog/otoearth/otospeech-task)
- [Public revision metadata](https://huggingface.co/api/datasets/otoearth/otoSpeech-full-duplex-task-oriented-20h)

## Candidate task: collaborative `Spot the Difference`

The publisher describes 18 sessions (6.31 hours) for this task and ten
author-created scene pairs. Each participant sees a different version of a
scene; neither sees the other participant's image. The speakers use audio as
the communication channel, and the release includes curated ground-truth
difference lists and timestamped interaction events. This gives a concrete
audio-plus-image/screen task with information asymmetry and a checkable task
outcome.

A prospective Decision target could classify, at a fixed interaction prefix,
which candidate scene difference the pair has confirmed or marked. The two
participant views and audio prefix would be model inputs. Event logs may be
used only to construct targets and cut points; feeding the target event or
future events to the model would leak the answer. Compare audio-only,
image-pair-only, and joint input, with wrong-session audio, swapped image
pairs, missing-input, and candidate-order controls. The dataset's own fixed
answer to “which differences exist?” is image-solvable by itself, so that
question alone would **not** establish audio's contribution.

## Unresolved admission gates

| Gate | Current evidence | Status |
|---|---|---|
| Repository revision and dataset-level license | Public API revision and dataset card state `CC BY 4.0` | Partially verified; underlying participant/stimulus terms and model-use application remain to be checked after access |
| Data access | HF files are gated and require sharing account information plus manual approval | Not authorized or requested |
| Task relevance | `Spot the Difference` has asymmetric views, audio interaction, and reference differences | Strong design candidate; no model necessity or quality demonstrated |
| Group-disjoint split | 18 sessions and ten scene-pair identities are reported; speaker/session overlap structure is not public in the card | Unknown; ten stimulus groups are too few for a broad quality claim |
| Independent evaluation | No permitted split manifest or untouched session list reviewed | Not established |
| Input modality | Per-speaker audio, screen/task stimuli, event logs; not camera video | Audio+image/screen only; does not satisfy video coverage |
| Local cost | 3.88 GB release; selected-file sizes and decoding requirements unknown | Unknown; do not download until access and a bounded subset are approved |
| Training/evaluation | No data opened, no features, model, or labels used | Not started |

## Next gate

Keep the source `REVIEW`. Before any download, request, or training, establish
permission to accept the Hugging Face access conditions and share the required
account information. If access is later available, inspect only the permitted
manifest/schema first, then verify speaker/session/scene-pair grouping and
filter to the author-created `Spot the Difference` stimuli. Stop if a
leakage-safe split with enough independent groups cannot be made. Any admitted
pilot must be labelled a small English collaborative screen task and must not
be presented as natural-video or general Omni quality.

No model, dataset file, or existing experiment was changed. The only work in
this screen was read-only public metadata review.

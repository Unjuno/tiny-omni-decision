# R2 public source refresh — 2026-11-22

## Scope

This was a metadata-only screen of two additional public repositories after the
current roadmap review found R2 M-2 still lacks an admitted task that requires
multiple streams. No dataset payload, model, validation/test annotation, or
sealed-audit material was downloaded or opened. The existing Perception Test
validation was not rescored; its prior use and the stop-tuning disposition in
[`R2_PERCEPTION_SOURCE_TASK_DISPOSITION_20261029.md`](R2_PERCEPTION_SOURCE_TASK_DISPOSITION_20261029.md)
remain controlling.

## Candidates

| Source (pinned repository revision) | What the publisher documents | Gate result |
|---|---|---|
| [CALVIN](https://github.com/mees/calvin/tree/fa03f01f19c65920e18cf37398a9ce859274af76) — `fa03f01f19c65920e18cf37398a9ce859274af76` | Simulated manipulation demonstrations with RGB/depth, robot state, actions, and language. The repository says its software is MIT; it does not establish that this license also covers every dataset artifact. Its documented debug archive is 1.3 GB; the D→D split is 166 GB, ABC→D is 517 GB, and ABCD→D is 656 GB. | `REVIEW_NOT_ADMITTED`. Dataset-specific rights are unresolved. It has no documented audio stream and its benchmark does not by itself establish that an option-selection target requires both vision and state. The debug archive would require a separate capacity check before any use. No download requested. |
| [Seeing Sound](https://github.com/sonyc-project/seeing-sound-dataset/tree/85a799afd0814b22648389853eabfc701e718396) — `85a799afd0814b22648389853eabfc701e718396` | CC BY 4.0; 60 synthesized 10-second audio soundscapes, generated ground-truth sound-event annotations, and crowdsourced annotations. The paper investigates waveform/spectrogram visualization as annotation aids. | `NOT_AN_AV_SOURCE`. The repository distributes audio and JAMS annotations, not synchronized video/image assets. Converting audio to a spectrogram would duplicate one stream rather than supply independent visual evidence. Sixty soundscapes also do not meet the roadmap's provisional 200 independent-group quality floor. No payload downloaded. |

## Decision

Neither candidate advances the multimodal task gate. Do not run a model,
extract features, or reinterpret the task to manufacture a joint-input claim.
Continue screening for a rights-cleared source with aligned inputs, a
defensible task where both streams matter, and enough independent groups. The
4-modality quality target and all product release gates remain open.

## Sources and limits

- CALVIN dataset split sizes and download options are from its pinned
  [dataset README](https://github.com/mees/calvin/blob/fa03f01f19c65920e18cf37398a9ce859274af76/dataset/README.md);
  the repository [license](https://github.com/mees/calvin/blob/fa03f01f19c65920e18cf37398a9ce859274af76/LICENSE)
  applies to code. No separate dataset grant was established in this screen.
- Seeing Sound modality, size, generated labels, and terms are from its pinned
  [README](https://github.com/sonyc-project/seeing-sound-dataset/blob/85a799afd0814b22648389853eabfc701e718396/README.md).
- This is a source screen, not a complete legal opinion or dataset-quality
  assessment.

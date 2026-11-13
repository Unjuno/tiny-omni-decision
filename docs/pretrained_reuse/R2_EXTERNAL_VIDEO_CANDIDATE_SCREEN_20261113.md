# R2 external video candidate screen — 2026-11-13

## Scope

Metadata-only check of two newly surfaced video / multimodal candidates against
the roadmap's product-directed source gate. No dataset payload, media,
annotations, labels, model, or evaluation split was downloaded or opened. This
screen does not admit a task or authorize fitting.

## Findings

| Candidate | Official task / modalities | Rights stated by source | Disposition |
|---|---|---|---|
| TUMTraf VideoQA | 1,000 traffic-scene videos and multiple-choice spatiotemporal video QA, plus grounding/captioning | Dataset page states CC BY-NC-SA 4.0 | Exclude from product-directed fitting unless a separate rights grant is established. |
| ActionSense | Synchronized video, audio, wearable/environment sensors and activity labels | Official project page states CC BY-NC-SA 4.0 and limits use to non-commercial purposes | Exclude from product-directed fitting. |

Primary sources: [TUMTraf VideoQA project page](https://traffix-videoqa.github.io/)
and [MIT ActionSense project page](https://action-net.csail.mit.edu/).

TUMTraf's task is video-only multiple-choice QA, so it would not close the
roadmap's both-input-required fusion gate even if rights were suitable.
ActionSense has synchronized modalities, but the stated non-commercial,
share-alike terms do not meet the current product-directed source requirement.
No license was accepted and no attempt was made to obtain restricted data.

## Next decision

Do not spend runtime or training effort on these candidates. Keep the already
audited Apache-2.0 VideoMAE V2 checkpoint as a feasible CUDA video encoder, but
require an independently grouped, rights-cleared Decision task before a new
quality fit. The existing VideoPrism CPU path and failed Torch-port parity
result remain separate evidence and are not overwritten by this screen.

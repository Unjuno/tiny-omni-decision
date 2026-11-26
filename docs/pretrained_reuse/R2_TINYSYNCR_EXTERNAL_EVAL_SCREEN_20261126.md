# TinySYNCR external-evaluation source screen — 2026-11-26

## Disposition

**Do not use TinySYNCR for training, checkpoint selection, or iterative tuning.**
It can only be considered as a public benchmark diagnostic after a video
Decision candidate is frozen; it is not blind. No dataset payload file or
video bytes were downloaded or opened in this screen.

## Pinned metadata

- TinySYNCR dataset: [CrossVideoReasoning/TinySYNCR](https://huggingface.co/datasets/CrossVideoReasoning/TinySYNCR).
- Observed Hub revision: `00decf9030eb24ad7df213d014ebc7b17c473188`
  (last modified 2026-05-06).
- The pinned card identifies 400 test items, 25 per task, and states that
  annotations and video assets are released under CC BY 4.0. Its dataset-viewer
  page rendered example questions and answers during this screen; those labels
  are treated as exposed. The set cannot serve as a validation or
  model-selection set, and a future score must be called public benchmark
  evidence rather than a blind audit.
- The pinned file-tree API listed 46 files, including 43 MP4s totaling
  51,274,272 bytes and 160,410 bytes of metadata. The Hub API's repository-wide
  storage field reported 575,896,459 bytes, while the current web UI displayed
  2.13 GB. The pinned tree sum is the relevant acquisition estimate; the larger
  repository/UI figures likely include data outside the pinned tree and are
  recorded as an unresolved metadata discrepancy.

## Relationship to full SYNCR

The full [CrossVideoReasoning/SYNCR](https://huggingface.co/datasets/CrossVideoReasoning/SYNCR)
revision observed during the same screen was
`e842ea1c7679a39f14c8a9b55b4ce0deeca648f6`, with approximately 49.1 GB of
stored data. Its card describes 15,960 training items and 4,000 test items
over disjoint video pools, but says only the annotations are CC BY 4.0 and
that rendered clips remain subject to upstream simulator and scene-asset
terms. Its upstream sources include CLEVRER, Kubric, and Habitat; the released
Habitat scenes may carry additional restrictions. This conflicts in scope
with TinySYNCR's broad CC BY statement for its bundled videos. The conflict
has not been resolved by item-level provenance or rights receipts.

The task is cross-video reasoning over two or more videos plus text, not
audio-video fusion. Its 25 examples per task are too small for robust
task-wise model selection. Even if the subset's CC BY grant is valid, its
public test labels make it suitable only for a clearly labeled one-shot
external diagnostic after candidate selection, with no follow-up tuning.

## Decision and next use

Do not download the full SYNCR bundle or use either source to train a readout
while the upstream-video license conflict remains unresolved. Preserve the
TinySYNCR revision only as a possible low-volume, public evaluation-only
diagnostic; before evaluating it, confirm exact video-path coverage and rights,
then freeze the candidate and scoring protocol. Do not use it as a blind audit
or tune after viewing its results. This screen does not close the R2
rights-cleared AV-fusion gate or establish video quality.

## Official references

- [Pinned TinySYNCR card](https://huggingface.co/datasets/CrossVideoReasoning/TinySYNCR/tree/00decf9030eb24ad7df213d014ebc7b17c473188)
- [Pinned full SYNCR card](https://huggingface.co/datasets/CrossVideoReasoning/SYNCR/tree/e842ea1c7679a39f14c8a9b55b4ce0deeca648f6)
- [SYNCR project repository](https://github.com/SaraGhazanfari/SYNCR)
- [Habitat-Lab dataset licensing notes](https://github.com/facebookresearch/habitat-lab/blob/main/README.md)

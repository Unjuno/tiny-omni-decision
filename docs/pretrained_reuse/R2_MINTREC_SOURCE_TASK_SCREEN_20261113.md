# MIntRec / MIntRec2.0 source and task screen — 2026-11-13

## Decision

**Do not admit for product-directed fitting.** The candidate is relevant to
text/audio/video intent decisions, but the dataset release is marked
CC BY-NC-SA 4.0 and the raw clips are collected from commercial television
series. No separate rights grant for the underlying audiovisual works was
identified. No data, features, media, annotations, or model were downloaded or
opened for this screen.

## Candidate fit and limits

The official MIntRec2.0 repository describes 15,040 utterances in 1,245
dialogues, with 30 fine-grained intent labels and text, video, and audio. It
also says the clips come from *Superstore*, *The Big Bang Theory*, and
*Friends*, with dialogues divided by scene and episode. The original MIntRec
paper describes 2,224 segments from 43 videos and says annotators selected
intent labels using video, audio, and text together. This makes the task
structurally relevant, but does not prove that every modality is necessary
for the frozen Decision task or that the splits are independent by source
episode, speaker, or scene.

The candidate would also require a careful group split and matched
text-only/audio-only/video-only controls. Those questions are moot for the
current product-directed path because the license and underlying-source
rights gates fail closed. The benchmark's intent taxonomy is a narrow social
intent task, not general multimodal Decision quality.

## Pinned primary evidence

- [Official MIntRec2.0 repository](https://github.com/thuiar/MIntRec2.0):
  modalities, dataset size, dialogue organization, and TV-series source.
- [Zenodo MIntRec2.0 release, version 3](https://zenodo.org/records/8041676):
  dataset record explicitly states CC BY-NC-SA 4.0.
- [Original MIntRec paper](https://hanleizhang.github.io/files/ACM-MM22-MIntRec/MIntRec.pdf):
  2,224 segments / 43 videos; annotators label intent by combining all three
  modalities; paper states CC BY-NC-SA 4.0.

No terms were accepted and no access-controlled source was attempted. Do not
reconsider unless an authoritative commercial-use grant covers both the
released annotations/features and underlying audiovisual media.

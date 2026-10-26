# RAVDESS audio-visual source screen — 2026-10-26

## Decision

Keep RAVDESS at **REVIEW / not admitted**. The official Zenodo record is a
task-relevant lead for a narrow acted-emotion comparison because each
performance is available as audio-video, audio-only, and video-only, and the
record includes 24 actor identities. However, the same Zenodo record contains
conflicting license statements: its description calls the release a Creative
Commons Attribution license, while its explicit license section identifies CC
BY-NC-SA 4.0 and says commercial licenses may be purchased. Do not fetch or use
the payload until the maintainers clarify the applicable terms. No request was
sent and no media or labels were downloaded or opened.

## Task and grouping fit

The official record describes 7,356 files (24.8 GB) covering 24 professional
actors, two lexically matched statements, speech and song, and several acted
emotion/intensity conditions. It provides audio-only, video-only, and
audio-video recordings. Filenames encode modality, channel, emotion, intensity,
statement, repetition, and actor, so any future corpus builder must strip these
fields from model inputs and split by actor before constructing paired records.
Statement/repetition grouping must also be checked to prevent near-duplicate
performance leakage.

This could support a **narrow acted vocal/facial-emotion Decision probe** with
matched unimodal and AV controls. It does not represent general world-state
reasoning, spontaneous affect, or the project’s full text/image/audio/video
target. Emotion labels appear in official filenames, requiring a leakage-safe
adapter and an audit of every metadata field before any feature extraction.

## Resource and model-selection limits

The complete collection is 24.8 GB, larger than the 10.95 GB free on C: at
this screen. The record lists individual video archives around 0.48–0.60 GB,
but a convenient archive size does not establish sufficient independent actor
groups or justify a subset. Any later study would need a predeclared actor-
disjoint train/development/calibration design, verify whether the pinned audio
and video encoders overlap this corpus, and avoid the already published
perceptual ratings as a blind audit unless a new independent cohort is
available.

## Sources

- [Official RAVDESS Zenodo record](https://zenodo.org/records/1188976), created
  2018-04-05, version 1.0.0; its page reports modification 2026-06-03.
- The record’s description says “Creative Commons Attribution”; the explicit
  License information section identifies [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
  This internal inconsistency is unresolved here.

## Next gate

Ask the maintainers to identify the operative license for the media and
annotations before considering acquisition. If terms are clarified, treat the
dataset as a narrow research probe only, with an actor- and statement-aware
split, no filename-derived input features, a source-overlap audit, and a
separate task-specific quality report. Do not infer product authorization or
general multimodal capability from this candidate.

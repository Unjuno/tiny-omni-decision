# AVQA source screen — 2026-10-19

## Disposition

The official Tsinghua AVQA source is a strong **task-design match** for
audio/video decision work, but it is **REVIEW / NOT_ADMITTED** for this
product-oriented training/evaluation path. The official terms grant personal
or classroom use and restrict commercial or profit-making use. The video
corpus is sourced from VGG-Sound YouTube clips, whose underlying upload rights
remain with the uploaders. No dataset annotations, video, audio, or extracted
features were downloaded or opened for this screen. No model run or metric was
produced.

## What the source offers

The official project describes 57,015 videos and 57,335 manually annotated
question/answer pairs. Its annotation schema includes multiple-choice options,
the correct option index, question type, and a `question_relation` field with
`View`, `Sound`, and `Both` values. That makes it a better semantic candidate
for a joint-input diagnostic than simple audio/video co-occurrence. The
project's motivating example explicitly requires combining a visible event
with an audio cue.

These source-level statements do not establish that every `Both` question
actually requires both inputs, that the distractors are calibrated for this
project, or that the source's train/test grouping meets our asset-disjoint
policy. Those checks would be required even if rights were resolved.

## Rights and data boundary

- The official AVQA site permits digital or hard copies for personal or
  classroom use, prohibits copies made or distributed for profit or commercial
  advantage, and says third-party component rights must be honored.
- The official AVQA site says raw videos are drawn from VGG-Sound and provides
  YouTube IDs, timestamps, and a train/test split. The underlying footage
  remains subject to uploader rights.
- No test annotations, challenge labels, or media were accessed.
- No source rows are approved for training, validation, calibration, or
  evaluation in the product-model pipeline by this screen.

## Next gate

Do not ingest this source into the product experiment. Revisit only if the
project's use is explicitly limited to the source's permitted non-commercial
research context and the video rights/data handling are independently
resolved. Any such research-only probe must stay separate from product claims,
must use asset-grouped splits, and must verify `Both` examples with matched
audio-only/video-only controls or counterfactual input removal. Continue
searching for a source with compatible rights before selecting another
real-world joint task.

## Sources

- [Official AVQA project page](https://mn.cs.tsinghua.edu.cn/avqa/), including
  task description, annotation schema, download information, and terms.
- [Official AVQA repository](https://github.com/AlyssaYoung/AVQA), including
  dataset overview and processing documentation.
- [VGG-Sound project page](https://www.robots.ox.ac.uk/~vgg/data/vggsound/),
  the stated video source.

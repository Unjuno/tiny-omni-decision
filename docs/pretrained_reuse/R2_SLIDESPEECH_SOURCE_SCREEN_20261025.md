# SlideSpeech source/task screen — 2026-10-25

## Decision

**Status: REVIEW; do not download, train, or evaluate.** SlideSpeech is a useful
task-design lead for audio-plus-slide recognition, but the source evidence
reviewed here does not establish permission to use the underlying recordings
for model training. It also does not yet supply a group-disjoint multiple-choice
Decision protocol that proves both modalities are necessary.

## Task fit

The official corpus description frames SlideSpeech as multimodal ASR and says
slides can provide domain-specific terms that help correct speech recognition.
The official benchmark reports contextual-ASR results with slide-derived OCR
and keyword bias lists. That suggests a practical candidate task: select the
intended technical term from acoustically plausible alternatives, where the
audio distinguishes pronunciation and the slide supplies domain context.

This is not already a Decision benchmark. The documented task is speech
transcription, not calibrated choice probabilities, and the public page does
not define controlled examples where audio-only and slide-only systems both
fail while the combination succeeds. Any future task conversion would need
predeclared candidates, answer construction, audio-only/slide-only/shuffled
controls, and splits grouped by original source video.

## Rights, access, and split evidence

- OpenSLR's official SLR144 entry states **CC BY-SA 4.0** for SlideSpeech and
  describes the corpus as collected from YouTube. It lists 1,659 videos,
  1,000+ hours, a 473-hour transcribed training portion, and dev/test portions.
- The project website scopes its CC BY-SA statement specifically to the
  **provided metadata**. Its official download instructions retrieve the
  recordings from YouTube using `yt-dlp`.
- The sources reviewed do not establish that the original YouTube audio/video
  is covered by the metadata license or separately cleared for model training.
  This unresolved media-rights gap is sufficient to keep the source out of the
  admitted training/evaluation pool.
- The public pages distinguish train/dev/test sets, but this screen did not
  verify video/source grouping, content overlap, or that the split is valid for
  a newly constructed Decision task. No archive members or annotations were
  accessed.
- The listed full train archives are large: 232.43 GB video and 88.56 GB audio;
  dev is 887.68 MB video and 429.96 MB audio; test is 1.45 GB video and 737.77
  MB audio. The project also lists about 1.37 GB of segments, transcript, and
  OCR metadata. No files were downloaded.

## What would change the decision

Reconsider only after the authors or rights holders provide an explicit grant
covering the specific media, model training, evaluation, derived weights, and
intended distribution; a concrete task protocol is frozen; source-video groups
and overlap are audited; and a small, legally admitted train/validation subset
is available. The metadata license alone is not treated as permission for the
underlying recordings.

## Primary sources

- OpenSLR SLR144 listing: <https://www.openslr.org/144/>
- SlideSpeech project site and metadata-license scope:
  <https://slidespeech.github.io/>
- Official download scripts and YouTube acquisition procedure:
  <https://github.com/Mashiro009/slidespeech_dl>

# KilometerAudio joint audio-video source screen — 2026-11-28

## Disposition

**RESERVE_ONLY; NOT ADMITTED FOR TRAINING OR ITERATIVE SELECTION.** KilometerAudio
is unusually well matched to a real joint audio-video Decision evaluation, but
the official release has no training split and the licensing/access terms for
the question package and underlying YouTube recordings are not established by
the page reviewed. Do not download the package, open test rows, or reconstruct
test videos while the candidate architecture and evaluation protocol are
unfrozen.

## What the official source establishes

The [official KilometerAudio page](https://perception-test-challenge.github.io/kilometeraudio.html)
describes an audio-video multiple-choice task over unedited urban walking-tour
videos, each one to three hours long, with five options. It says human raters
collected the questions and the research team verified that they require both
audio and visual evidence. The declared task families include locating sounds
relative to visible landmarks, time-localizing audio events, counting/multi-hop
audio-visual reasoning, and recognizing speech or music in context.

The official page lists 508 questions over 221 unique videos: 8 validation
questions on 6 videos and 500 test questions on 218 videos. It states that
validation reuses audio questions from the 2024/2025 hour-long QA benchmark,
whereas all 500 2026 test questions are new and only 3 of the 218 test videos
appeared in earlier editions. **The validation is therefore not a clean new
selection set.** The page lists no train split. A 54 KB dataset package is
linked, but its contents and whether it exposes answers were not inspected.

## Data-quality strengths and limitations

- Strong task design for audio-video complementarity: the authors explicitly
  describe human verification that both modalities are needed.
- New test questions and mostly new test videos relative to prior challenge
  editions provide a potential locked external assessment cohort.
- The 218 test video groups are a small effective sample for estimating broad
  task quality, despite 500 questions; uncertainty must be clustered by video.
- Hour-long inputs create a substantial temporal-retrieval burden and may
  require indexing. A result from a small frame-sampling policy would not be a
  fair full-video assessment.
- The page does not establish commercial-use rights for the benchmark
  annotations, a license for the 54 KB question package, per-video reuse
  rights, consent/publicity scope, or participant/source-group splits beyond
  the counts shown. The videos are YouTube walking tours; original uploader
  rights and platform access terms remain relevant.
- The challenge page reports system accuracies, including an agentic best
  result, but these are not results for this repository's models and must not
  be treated as our quality evidence.

## Next gate

Keep KilometerAudio unopened as a possible **one-time, locked joint AV audit**,
not training data. Before any access:

1. Confirm annotation and video-use terms for the intended product-directed
   work, including the official challenge server's evaluation conditions.
2. Confirm whether the linked package includes test labels or other answer
   material; if so, it cannot serve as a blind audit.
3. Freeze the candidate model, preprocessing, temporal coverage, abstention
   policy, and scoring code before any test access.
4. If admitted, report video-clustered metrics and compare audio-only,
   video-only, missing/shuffled/offset controls only if the protocol permits
   those controls without using test labels for selection.

The official challenge report states the best generalist single-model result
and agentic results separately; none is directly comparable with the compact
Path A models or a release target. This screen used public page-level
information only. No question package, test row, video, or answer key was
downloaded or opened; no model, GPU, training, or sealed audit was used.

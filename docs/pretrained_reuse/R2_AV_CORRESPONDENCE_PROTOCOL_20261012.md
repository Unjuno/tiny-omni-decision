# Prospective audio-to-video correspondence probe

## Scope and motivation

The visibility proposal failed its query-only control (96.44% without media).
Do not fit it. The next bounded candidate asks which of four silent video
observations originates from the query audio's recording. No sound-class label,
ground-truth object identity, visibility, filename or recording ID enters the
scorer. Correctness is recording correspondence, not a claim that all sounds
are visually identifiable or that recordings cannot sound alike.

This is a task-matched AV alignment diagnostic within R2-M. It does not replace
the required general text/image/audio/video decision battery, nor demonstrate
that sensors or optional natural-language conditions are solved.

## Freeze before acquisition and fitting

- Source: official Perception Test archive generations, pinned by manifest.
  Retain CC-BY attribution and exact member hashes. No test/audit archive.
- Development acquisition ceiling: 256 official-train and 200 official-valid
  recordings, selected by SHA-256(`17:video_id`) ascending, independent of labels.
  First smoke: eight train recordings only. Previously fetched matching files
  may be reused after hash verification. Never substitute shorter/easier clips.
- Keep official split assignments; upstream groups script × participant, not
  participant alone. Do not random-split official train into validation.
- Within each split, partition the fixed ordered IDs into quartets. Query all
  four recordings' audio against the same four videos. Each video is correct
  exactly once per quartet. Assign each of the four correct positions once per
  quartet using seed17-derived hashes; order distractors separately per query.
  Score content only. ID strings remain routing metadata.
- Failed decoding, missing sound and semantically ambiguous correspondence stay
  explicitly reported. No quality-based exclusion or audio class balancing.
- Video: existing eight-frame frozen VideoPrism profile; audio: frozen DyMN.
  Use the common observed audio/video interval and actual timestamps; no labels
  or trailing answer-reveal frames. Resolve cut handling before any scored run.
  No hidden per-clip resampling or encoder changes based on outcomes.
- Query audio and candidate videos are separate observations. Neither metadata
  identity nor candidate position may be provided to the learned score function.

## Required controls and interpretation

Untrained fixed projections, train-fitted projection, audio-zero, video-zero,
shuffled-query audio, reversed candidate order and one-to-one ID/hash checks.
Chance for the balanced four-candidate construction is 25%; observed control
scores and calibration must still be measured. Balance does not rule out
recording/environment shortcuts. Same sound classes and repeated scripts may
make mismatched recordings acoustically plausible; preserve these cases.

First fitting candidate, only after the acquisition/rights/timing gates pass:
seed17, frozen encoders, 64-dimensional bias-free projections for 384D audio and
768D video, normalized dot-product logits with fixed temperature0.1. AdamW
lr0.001, weight decay0.01, batch of quartets, 100 epochs. Select earliest minimum
validation NLL, retain final separately. No extra seeds, schedules or capacity
changes in this initial comparison. A smoke may check mechanics on train only;
training accuracy from that smoke is not quality evidence.

Report Accuracy/NLL/Brier/ECE, source recording counts, quartet-grouped bootstrap
10,000 draws seed17, all attempted configurations, best/final reload, latency,
memory and total resident parameters. Quartets couple candidate errors and are
the minimum resampling unit; participant overlap remains an additional limitation.
This protocol does not authorize claiming 95–98% general quality from retrieval.

## Current status

Protocol fixed before new media acquisition or readout fitting. Source admission
and media/cut handling are still required. Existing two-video feature artifacts
are preserved; their event-level visibility windows are not silently repurposed
as recording-level retrieval examples. Do not train or score this task until the
correspondence records pass the stated gates.

## Executed manifest construction

The deterministic builder produced 256 train routing records in 64 quartets from
the pinned train annotation keys. Reversing input enumeration leaves the result
unchanged. Tests verify exact positive identity, identical candidate sets within
each quartet, balanced correct positions and rejection of test split requests.
The JSON SHA-256 is
`560f3bc7235c842746038e6e5a886471f4dc08cc515d9eba4d12968e608f393b`,
stored at `perception-correspondence-plan-01/train-routing.json` under the R2 root.
This file contains supervision/provenance and must not be embedded as model text.

The first smoke IDs are video_3174, video_7855, video_9027, video_1688,
video_4873, video_1693, video_2797 and video_5013. No validation IDs or labels were
inspected to choose them. Only eight-clip acquisition is initiated in this stage;
the remaining planned train records do not imply downloaded media or training.

# TartanAviation multimodal task screen — 2026-11-14

## Decision

**Status: `REVIEW_NOT_ADMITTED`.** TartanAviation is a promising temporal
audio/video/trajectory source, but the public metadata reviewed here does not
establish a usable supplied-option Decision target, aligned utterance-to-flight
labels, or a leakage-safe independent group split. Do not acquire media, build
a corpus, or fit a model from this screen alone. No media, weights, annotations,
or evaluation labels were loaded in this screen.

## Pinned source and rights

- Official repository: <https://github.com/castacks/TartanAviation>.
- Reviewed revision: `077a04abac1e50953c45b72524edd79e3a21d9bd`.
- The pinned repository README identifies the dataset (vision, speech, ADS-B,
  and weather) as CC BY 4.0 and the repository code as BSD-3-Clause. This is
  source-level rights evidence; it does not settle privacy or downstream use
  questions for aircraft identifiers.
- Official dataset page: <https://theairlab.org/tartanaviation/>.
- Paper: <https://doi.org/10.1038/s41597-025-04775-6>.

## Metadata findings

The official README describes approximately 3.1 million images in 550 KAGC
videos, 3,374 hours of ATC speech in 677 daily archives, and 661 days of ADS-B
trajectories. The listed download sizes are about 3.5 TB for vision, 543 GB for
filtered speech, and 2 GB for ADS-B. The metadata-only `vision/weather_stats.csv`
at the pinned revision is 35,369 bytes, with 550 rows and 550 distinct video
names. It reports five snow videos and 28 rain videos; the five snow videos and
all videos with visibility of 2 miles or less are concentrated on one morning,
2022-11-18. Thus a random video split would not establish independent weather
generalization.

The official page states that each video sequence has synchronized ADS-B and
aircraft labels. Speech is organized in daily archives with per-clip start/end
time metadata. The paper describes separate trigger thresholds for camera and
speech capture (8 km and 10 km, respectively), so co-location at the airport
does not by itself prove clip-level speech/video alignment or that an audio
utterance names the aircraft represented by a video label. The public README
reports 239 labeled aircraft and says 72 occur in more than one video. A valid
split may therefore need to group by aircraft identity as well as sequence;
the component count after joining repeated aircraft to sequences is unknown.

## Task and split feasibility

A possible task is selecting which opaque aircraft option matches an ATC
utterance and a short visual/trajectory observation. It could exercise
temporal evidence and cross-modal grounding. However, the reviewed metadata
does not provide verified utterance transcripts or a direct utterance-to-track
Decision label. A target would require a separately justified annotation or
alignment procedure. Callsigns, ICAO addresses, N-numbers, filenames, or task
metadata must not become answer-bearing model inputs.

Only 550 video sequences are listed, all from one airport and fixed camera
installation. That is a limited upper bound on video-level independent groups;
it is not evidence of 550 independent scenes or aircraft groups. The exact
aircraft/sequence connected-component count, a fresh exposure inventory, and
the number of examples remaining after identity-disjoint partitioning are
unknown. No split or evaluation set is admitted.

## Disposition and next gate

Keep this candidate in review. Reconsider only after a metadata-only feasibility
pass establishes (1) a reproducible, manually or author-verified target for
joint audio/video/trajectory choices, (2) source-exposure accounting, (3) a
deterministic split grouped by both aircraft identity and video/event identity,
and (4) sufficient independent groups in every required partition. Estimate
the intended subset's actual download and feature-storage cost before any
acquisition. If those gates fail, close this candidate without downloading the
multi-terabyte vision collection.

## Sources

- Pinned official README: <https://raw.githubusercontent.com/castacks/TartanAviation/077a04abac1e50953c45b72524edd79e3a21d9bd/README.md>.
- Pinned repository code license: <https://raw.githubusercontent.com/castacks/TartanAviation/077a04abac1e50953c45b72524edd79e3a21d9bd/LICENSE>.
- Official dataset page: <https://theairlab.org/tartanaviation/>.
- Scientific Data paper: <https://doi.org/10.1038/s41597-025-04775-6>.

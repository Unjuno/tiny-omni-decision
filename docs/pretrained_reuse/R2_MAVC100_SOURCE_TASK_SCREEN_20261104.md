# MAVC100 source/task screen — 2026-11-04

## Disposition

`REVIEW_NOT_ADMITTED`. MAVC100 has a task shape relevant to paired audio/video
decisions, but its project-specific data terms conflict with the catalog's
CC-BY-4.0 claim. The project says the clips are copied from internet streams,
restricts the data to research use, and reserves annotation rights. The number
of clips is not evidence of independent source groups. Do not download the
archives, extract labels, train, or evaluate this candidate until the data
rights and source-group provenance are resolved.

## Pinned source evidence

- Official code/data repository revision inspected: `f14749de7116247b097cef070ae7d32491170fc3`.
- The project README describes two four-class label streams: audio-only events
  and audiovisual events where Speech/Singing refer only to the visible
  “anchor.” It says the released label files mark Speech, Singing and Silence;
  `Others` is inferred from remaining time. This could support a future
  frame-level anchor-vocalization Decision task, but is not yet an admitted
  evaluation protocol.
- The project-specific data README says there are 200 live-video clips of 30
  seconds each, that source videos came from the internet, and that the dataset
  is “only” for research use. It also reserves rights to annotations and asks
  for removal if source material infringes.
- The official project webpage describes MAVC100 as an open dataset. The NLM
  catalog lists CC BY 4.0. These statements do not resolve the narrower,
  conflicting restrictions in the data repository's own README. The root
  repository's MIT license covers code; it does not override the data-specific
  terms.
- A metadata-only, blob-filtered repository clone was created outside Git at
  `C:\CodexArtifacts\pretrained-reuse-r2\mavc100-source-screen-20261104-attempt-01`.
  No media archive, annotation archive, source URL list, or model was opened or
  extracted. No media download, fit, inference, validation, or audit access
  occurred.

## Gate assessment

| Gate | Evidence | Status |
| --- | --- | --- |
| Paired audio/video task is defined | Separate audio and anchor-conditioned audiovisual frame labels | Promising task shape |
| Rights permit this project's intended use | CC-BY catalog entry conflicts with repository's research-only and rights-reserved language; raw streams originate elsewhere | **Unresolved** |
| Independent groups | 200 clips are reported, but source-stream / performer grouping and cross-clip identity are not audited | **Unverified** |
| Enough held-out groups | No train/calibration/development split or independent group counts were established | **Unverified** |
| Label integrity and timing | Annotation archives were not opened; no frame/audio synchronization audit | **Unverified** |
| Product-ready training source | The project's own page restricts use to research | **Not admitted** |

## Next action

Prefer a source whose own terms clearly permit the intended model use and whose
recording/participant groups support a prespecified split. Reconsider MAVC100
only after the conflicting terms and grouping evidence are resolved. This screen
does not change any existing R2 task disposition, establish a quality result,
or justify a model architecture change.

## Primary sources

- [Pinned data README](https://github.com/Yuanbo2020/Audio-Visual-VAD/blob/f14749de7116247b097cef070ae7d32491170fc3/Open_dataset_MAVC100/README.md)
- [Pinned source-video note](https://github.com/Yuanbo2020/Audio-Visual-VAD/blob/f14749de7116247b097cef070ae7d32491170fc3/Open_dataset_MAVC100/Backup_of_source_video_streams/README.md)
- [Official MAVC100 project page](https://yuanbo2020.github.io/Audio-Visual-VAD/)
- [NLM dataset catalog entry](https://datasetcatalog.nlm.nih.gov/dataset?q=0002226000)

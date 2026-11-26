# Fresh multimodal task-source screen — 2026-11-26

## Decision

No newly checked source is admitted for a quality experiment in this screen.
The best task match, HWU-USP, has only 16 participant groups / 144 activity
recordings and its official Dryad file endpoint returned HTTP 403 in three
small ranged requests. No media was downloaded or read. Its metadata supports
a future exploratory pilot only if the official download becomes available;
it cannot pass the roadmap's provisional 200 independent-scene/clip floor.

This screen does not change the sub-SEVIR result or reopen any exposed
validation. It adds no claims about general multimodal quality.

## Candidates inspected

### HWU-USP activities (Dryad)

- Official [Dryad record](https://datadryad.org/dataset/doi:10.5061/dryad.v6wwpzgsj)
  lists an 8.13 GB archive. The source paper describes synchronized robot
  RGB-D video, wrist/waist IMUs, and ambient home sensors, plus nine high-level
  daily-activity labels. It reports 16 subjects and 144 complete recordings.
- The paper reports ethics-board approval, informed consent, and anonymization.
  Dryad's official [reuse policy](https://dev.datadryad.org/help/guides/reuse)
  says its datasets are CC0. The paper also describes a highly controlled,
  single-kitchen setting, so participant-held-out results would remain a narrow
  task test.
- The official file links list an 8.13 GB dataset archive and a 1.24 MB
  README. Direct 1 MiB range probes against the archive URL
  `https://datadryad.org/downloads/file_stream/568408` returned HTTP 403,
  118-byte error bodies, in 0.738 s, 0.505 s, and 0.536 s. No throughput
  estimate is inferred from these denied requests. No alternate route was
  attempted.
- Disposition: metadata-only `REVIEW_NOT_ADMITTED`. The 16 participant groups
  are below the provisional 200-group quality floor; 144 clips do not become
  144 independent subjects. The source is a promising architecture pilot,
  but no sufficiently powered task result can be claimed from it.

### MMCSG (Meta)

- Meta's official [dataset page](https://ai.meta.com/datasets/mmcsg-dataset/)
  describes synchronized conversation audio, video, and IMU, with consented
  participants and blurred faces.
- Its official [download agreement](https://ai.meta.com/datasets/mmcsg-downloads/)
  requires organization/affiliation and organization email, explicit
  acceptance, and grants a limited non-transferable license. It permits model
  development for research and commercial purposes, but prohibits incorporating
  the dataset into another dataset or product and includes deletion/audit
  obligations.
- Disposition: not accessed or requested. It is not suitable for this product
  task without a separate user decision and license review; no organization
  details were submitted and no agreement was accepted.

### KAIST rotating-machine multimodal data (Mendeley Data v6)

- The official [Mendeley record](https://data.mendeley.com/datasets/ztmf3m7h5x/6)
  lists CC BY 4.0 and aligned acoustic, vibration, temperature, and current
  time series. The primary [Data in Brief article](https://pmc.ncbi.nlm.nih.gov/articles/PMC10036499/)
  describes one laboratory rotating-machine setup. Its varying-load section
  covers multiple vibration/current/temperature conditions, but the acoustic
  portion lists only five conditions at 0 Nm.
- Disposition: metadata-only `REVIEW_NOT_ADMITTED` for fusion quality. The
  available acoustic coverage cannot support a balanced load-held-out
  multimodal comparison, and the primary record does not establish 200
  independent machine/session groups. No files were downloaded.

## Reproducibility and machine impact

- The three Dryad probes wrote only their 118-byte HTTP error responses under
  `C:\CodexArtifacts\pretrained-reuse-r2-active\hwu-uspspeedprobe`.
- C: had 114.24 GiB free before the probes. No model/GPU work, data archive,
  existing run, checkpoint, or sealed audit was touched.
- The currently measured sub-SEVIR Path A remains the only new R2 quality run
  in this worktree. It scored 89.62% on its task-specific validation and did
  not establish a statistically clear joint-input gain over radar-only.

## Next evidence gate

Continue metadata-only source screening for a naturally recorded,
rights-admitted video-plus-sensor task with enough independent source groups
and a clear reason both inputs can help. Do not promote the HWU-USP or KAIST
sources to a quality gate based on the small/group-limited evidence above. A
new BERT-series run is not indicated by this screen: text-only model selection
cannot resolve the missing joint multimodal task, and the roadmap already
records the tested BERT results as task-specific and mixed.

# MUVS video + mobile-sensor source/task screen (2026-10-28)

Status: **promising task shape; not admitted for data use or fitting. No files or media were downloaded.**

## Source and pinned release

- Official dataset record: [MUVS v1.0, Zenodo DOI 10.5281/zenodo.20708683](https://doi.org/10.5281/zenodo.20708683), published 2026-06-15.
- Publisher project page: [SWPERFI MUVS Dataset](https://swperfi-project.github.io/Pages-dev/MuvsDataset-page/).
- Zenodo lists a 9.7 MB `data.zip` (MD5 `8fd1d426e0b4308655ea71884d1b0c8e`), a 5 kB download script, README, and requirements. The page's Zenodo Rights/License field is blank; the project page inspected here does not state a dataset license or participant/publicity consent terms. Treat data rights as `UNKNOWN`; public accessibility is not permission.
- The project page says the period-video payload is hosted separately on public GCS and totals about 164.9 GB. It supports selecting specific events, but no per-event byte inventory was inspected. The Zenodo record's displayed data-volume statistic is distinct from its listed files and does not resolve the external payload size.

## Task fit and unresolved integrity

The publisher describes 17 live sports events (basketball, soccer, hockey,
volleyball), 5,039 aligned 3-second fragments, 258 period videos, and
fragment-level expert camera-selection labels. Each phone records video and 18
sensor channels (accelerometer, gyroscope, magnetometer, orientation, light,
pressure, proximity, GPS). This could map to a useful supplied-option Decision:
choose the best available camera/view for a fragment.

This is only a task proposal. Evidence still required before R2-M admission:

1. Explicit rights/license and consent/privacy scope for the source recordings, labels, and sensor streams. The project page reports face blurring but that alone does not resolve rights or consent.
2. Whether the train/test split is event-disjoint. The project page says events are distributed across both splits, which does not establish independent event groups; exact event IDs and camera/period-parent relationships must be audited.
3. Whether both video and sensor signals are needed. The expert's best-camera target may be predictable from video-only or annotation metadata. Freeze controls for metadata-only, video-only, sensor-only, and paired inputs, plus candidate-order shuffle and sensor/video mismatch before any fit.
4. A legally and storage-feasible subset with immutable file hashes and zero group/content overlap. Full video is far beyond current disk capacity; no event-size estimate or acquisition attempt was made.

## Disposition

Keep MUVS `REVIEW`. Do not download `data.zip`, run its downloader, fetch GCS media, fit a model, or report quality until the provider's dataset terms are explicit and the group-split/input-necessity preflight is approved. If rights are clarified, first request or inspect only a metadata-only event manifest and exact selected-event sizes; the current 0.82 GiB C: headroom is not sufficient to assume media acquisition is safe.

No model, code, existing artifact, or running process was changed. No GPU was used.

## Evidence sources

- [Zenodo v1.0 record](https://zenodo.org/records/20708683): listed archive, release date, dataset description and current Rights/License section.
- [Publisher project page](https://swperfi-project.github.io/Pages-dev/MuvsDataset-page/): modalities, camera-selection label, splits, event counts, event-selection downloader and stated full video size.

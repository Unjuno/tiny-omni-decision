# A3CarScene audio-video source and task screen — 2026-11-16

## Disposition

`REVIEW_NOT_ADMITTED`. A3CarScene is an attractive synchronized road-scene candidate: the paper describes audio, two dashcams, aligned timestamp labels, and road/weather/lighting/pavement conditions. However, the OSF project API reports no dataset license, and a conservative grouped split by collection day yields only 14 groups. The paper does not establish a task requiring both audio and video. Do not access the media or fit a model under this source as currently documented.

This was a metadata-only screen. No audio/video, annotation CSV contents, model, or evaluation data was downloaded or opened. The OSF API node and file metadata were read; the open-access paper was reviewed.

## Source and revision evidence

- Dataset project: [A3CarScene, OSF project ajxu6](https://osf.io/ajxu6/), public node created 2023-03-21 and last modified 2025-03-02.
- OSF node API: <https://api.osf.io/v2/nodes/ajxu6/>. At the check, `public=true` and `attributes.license=null`. The 20,675-byte UTF-8 node response had SHA-256 `c7a96b70dfb300bd622315fd28b2e9ae975ebb2c919eb58e819289708ad03ecb` (checked 2026-11-16 01:11:57 UTC).
- Publisher paper: [Cantarini et al., Data in Brief 48 (2023), 109146](https://www.sciencedirect.com/science/article/pii/S2352340923002652), DOI `10.1016/j.dib.2023.109146`. The article is open access under CC BY; this is the article license, not an OSF data license.
- Paper’s data-availability link: <https://osf.io/ajxu6/>.

## Metadata-only inventory

The paper describes approximately 31 hours per device, collected across 14 days in October–November 2022 and about 1,500 km. It reports 400 recording files (320 audio and 80 video), eight microphones and two dashcams, with timestamped labels for road type, urbanization, weather, lighting, pavement type/wetness, and window state.

The OSF `osfstorage` metadata folder listing contains 15 CSV file entries (one aggregate metadata file plus 14 dated metadata files; 144,352 bytes in total). The listed files include SHA-256 values in the OSF API. Their contents were not fetched or opened. The project also lists an external OneDrive provider node, but its content and access rights were not resolved in this screen. No media archive appeared in the inspected OSF metadata-folder listing; this does not establish that no separate media endpoint exists.

## Admission gates

| Gate | Result | Evidence |
|---|---|---|
| Dataset reuse rights | Fail / unresolved | OSF `attributes.license` is null. CC BY on the publication does not grant dataset reuse rights. |
| Synchronized AV and target labels | Pass at metadata level | The paper describes aligned audio/video plus road-scene labels. |
| Independent source groups | Fail for the provisional quality gate | Four hundred files are from one instrumented vehicle over 14 acquisition dates. Grouping by date gives at most 14 known independent units; a finer route/session identity is not present in the inspected file listing. The roadmap's provisional 200-group floor is unmet. |
| Both-input-required Decision task | Not established | The paper describes audio-only, video-only, and multimodal scene understanding as possible; it does not show a fixed target for which both streams are necessary under matched unimodal and perturbation controls. |
| Privacy/provenance | Partly described | The article says faces and license plates were censored. The dataset's own usage rights and any participant/driver consent scope were not verified. |
| Acquisition/fitting | Not performed | No media, label contents, checkpoint, or model was accessed. |

## Reconsideration criteria

Reconsider only if the dataset custodians publish or provide explicit terms covering the media and annotations, an auditable collection-session/route identity and a permitted group-disjoint split with sufficient independent groups (or an explicitly approved low-power exploratory protocol), and a fixed task with evidence that both audio and video contribute beyond best-unimodal controls. Until then, retain this screen as a rejected source lead and do not infer data rights from the paper's license.

## Reproduction metadata

- Screen date: 2026-11-16 (JST).
- No repository source commit applies to the external OSF record; the exact API endpoint, observed node revision dates, response size, and response SHA-256 are recorded above.
- No payload download, label read, training, evaluation, or GPU use occurred.

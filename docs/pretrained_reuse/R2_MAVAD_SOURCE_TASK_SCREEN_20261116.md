# MAVAD audio-video source and task screen — 2026-11-16

## Disposition

`REVIEW_NOT_ADMITTED` for this project’s grouped, both-input-required Decision evaluation. The dataset record itself declares CC BY 4.0, so the initial license concern is resolved for the deposited dataset files. The source still does not establish an auditable independent-group split or a task whose labels require both audio and video. Do not acquire the 12 GB media archives or fit a model under this proposal.

This is a metadata and official-paper screen only. No archive, media, label payload, model, or evaluation data was downloaded or opened. The Zenodo record metadata was queried through its JSON API; the paper and dataset metadata PDF were read through their public pages.

## Pinned source evidence

- Dataset: [MAVAD, Zenodo record 7950008](https://zenodo.org/records/7950008), DOI `10.5281/zenodo.7950008`, published 2023-06-01, version v1.
- Dataset record API: <https://zenodo.org/api/records/7950008>. Observed `access_right=open`, `metadata.license.id=cc-by-4.0`, and four files. The three media archives total 11,978,755,278 bytes by the API sizes below; the record page reports 12.0 GB. The API response was 6,509 UTF-8 bytes when checked at `2026-11-16T01:01:46.1023383Z`, SHA-256 `05c52e278affd7e94b50f4d4851723eaa9d13d6fe90adc7b93c34ff00ede202f`. The license applies to the deposited files as declared by the record; it does not establish a separate license for the code repository.
- Official dataset metadata: [MARVEL_AV_AD_Metadata.pdf](https://zenodo.org/records/7950008/files/MARVEL_AV_AD_Metadata.pdf), MD5 `e70d6dd2eeeb03936bd9fb9038d807fd`, 100,307 bytes.
- Associated paper: [Leporowski et al., arXiv:2305.15084v1](https://arxiv.org/html/2305.15084), submitted 2023-05-24.
- Paper-linked code: <https://gitlab.au.dk/maleci/audiovisualanomalydetection>. The current endpoint redirected to a service-information page, so repository contents and its license could not be checked. No code was fetched.

### Archive metadata from the Zenodo API

| File | Bytes | Zenodo MD5 |
|---|---:|---|
| `Masked_Mgarr.zip` | 5,498,240,496 | `b47ce8fe6a93d25ae6213410580edb78` |
| `Masked_Zejtun_Field.zip` | 4,659,232,563 | `94fb5d37c362d82ef14785ac6944c8a6` |
| `Masked_Zejtun_Scrapyard.zip` | 1,821,282,219 | `3cca413c94ffc5b6800aa6daa26fafae` |
| `MARVEL_AV_AD_Metadata.pdf` | 100,307 | `e70d6dd2eeeb03936bd9fb9038d807fd` |

## Task and grouping assessment

The paper describes a plausible narrow synchronized audio-video task: weakly supervised video-clip normal/anomaly classification, with 11 labels and audio/video streams synchronized at the camera. It reports positive audio contribution in each of the three camera scenes, although the magnitude differs: AUC rises from 87.91 to 89.76 in Mgarr, 63.18 to 64.10 in Zejtun Scrapyard, and 76.58 to 80.45 in Zejtun Field for the focused audio path. This is encouraging feasibility evidence, not proof that both streams are necessary for a supplied-option Decision task.

The split design does not meet the project's independent-group requirement:

- The complete dataset has 764 clips across exactly three fixed camera scenes: Mgarr, Zejtun Scrapyard, and Zejtun Field. The official paper repeats random 60/20/20 clip splits three times *within each scene*. That design evaluates clips from the same camera/location in train, validation, and test; it does not test generalization to independent scenes or cameras.
- The metadata PDF says annotators received five-minute video snippets and trimmed shorter labeled clips. Public metadata does not map each trimmed clip to its parent snippet or source recording. Therefore, repeated-parent/source overlap cannot be audited without inspecting archive contents, and clip IDs alone cannot safely be treated as independent asset groups.
- Grouping by the only confirmed scene/camera identity yields three groups total, far below the roadmap's provisional 200 independent scenes/clips floor where available. Treating clips as independent would assume away the unresolved parent identity and camera correlation.
- The paper frames the task as weakly supervised binary anomaly classification or 11-way clip-label prediction. It reports an audio contribution, but there is no predeclared both-input-required target, paired option-choice contract, or missing/shuffled/stale/timing-offset/contradictory-input intervention protocol for this repository.
- The paper states video faces/license plates were blurred; it also explicitly says audio was kept unmodified because the authors considered speech/PII unlikely at microphone distance. That statement is not audio anonymization or a consent audit. It remains a privacy limitation for any future use even though the dataset record declares CC BY 4.0.

## Gate result

| Gate | Result | Evidence |
|---|---|---|
| Deposited dataset license | Pass for the dataset record | Zenodo API declares CC BY 4.0. This does not resolve code license or audio consent. |
| Synchronized audio/video | Pass, at a broad dataset level | Paper describes three AV cameras and paired streams. |
| Independent split/grouping | Fail | Only three confirmed camera-scene groups; official random splits are within each scene; parent snippet IDs are not available in public metadata. |
| Both-input-required Decision target | Fail / not established | Published weakly supervised anomaly task shows an AV benefit but does not establish that both inputs are necessary under the project's controls. |
| Privacy/consent scope | Unresolved | Video masking is described; audio is unmodified and no consent evidence was found in the screened record. |
| Acquisition or fitting | Not performed | No archive, media, labels, code, model, or evaluation records were loaded. |

## Reconsideration criteria

Reconsider only if the source custodians provide or publish (1) parent-recording/session identities sufficient to enforce group-disjoint splits with at least 200 independent groups, or an explicitly approved lower-powered exploratory protocol; (2) an admissible both-input-required target and fixed matched-unimodal/missing/shuffled/timing-offset controls; and (3) clarification of audio privacy/consent and code licensing if the code is to be reused. Keep all future splits grouped by parent recording and camera scene. Do not use the paper's random within-scene test splits as a blind audit.

## Reproduction metadata

- Screen date: 2026-11-16 (JST).
- Zenodo record DOI: `10.5281/zenodo.7950008`.
- Dataset record page and paper were checked through official publisher/repository pages; the API fields and file checksums above were captured directly from the public Zenodo API.
- No repository source commit applies because this is a metadata-only publisher record. No code was downloaded.

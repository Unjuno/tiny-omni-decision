# R2 iNaturalist paired image-audio metadata screen — 2026-11-10

## Decision

Do not admit this candidate to Path A training or evaluation. The live API
screen returned 4,410 potentially paired observations, but after excluding
every observation/photo/observer identity found in the preserved local
iNaturalist inventories, only 109 records across 70 observer groups remained.
That is below the provisional R0 floor of 1,000 examples and 200 independent
groups. The validation portion has only 17 examples across 10 observers.
Repeated windows or records from the same observers cannot satisfy the group
floor. No image/audio media, feature cache, model output, or sealed audit was
opened; no model or GPU was run.

This is an image-plus-sound source, not a video source. iNaturalist's official
help says video uploads are not supported. The surviving pool also has not
passed a frozen task/control design demonstrating that both modalities are
needed for the same decision. Consequently, even a larger count would require
that task-fit gate before media acquisition or model evaluation.

## Reproducible metadata screen

- Source: public iNaturalist API v1 observations endpoint.
- Query: `taxon_id=3` (Aves), research-grade, photos and sounds required,
  observation/photo/sound licenses each `cc0`, `per_page=200`.
- API result count: 4,410 across 23 pages; screen completed
  2026-11-09 19:10:24 UTC.
- Historical identity inventory: 46,529 rows in 59 local JSONL files,
  including 14,967 observations, 14,967 photo IDs, 2,746 observer-group
  hashes, 9,058 media hashes, and 4,529 normalized image hashes. No historical
  sound IDs were recorded.
- Historical overlaps removed: 4,301 rows. Duplicate observation rows in
  the candidate result: 0.
- Remaining candidate: 109 records; 92 train records / 60 observer groups and
  17 validation records / 10 observer groups; 92 distinct taxa. Train and
  validation have zero observer, observation, photo, or sound ID overlap;
  all retained observation/photo/sound licenses are CC0.
- Split identity uses hashed observer groups; observation and photo identity
  exclusions were applied before candidate retention. The per-row candidate
  manifest contains no usernames, coordinates, descriptions, or media URLs.
- Candidate manifest SHA-256:
  `9e0a4391adcd98947da6f08f6cb7f0b77578a4f3bcc6f8b55c107a05614766c7`.
- Screen summary SHA-256:
  `9dc0ae2072454250a007e31f1bc79b8b2a2800c13ae7042a4da38ca86e45e245`.
- Historical identity-set SHA-256:
  `7616ae4a6a07311808cd31b37a0e7834eca3f7da0f15a7474fa45bcfad71458e`.
- Local evidence directory:
  `C:/CodexArtifacts/inat-cc0-paired-image-audio-screen-20261110-attempt-02`.
  Attempt 01 is an earlier incomplete connection attempt and was left intact.

The API query and counts are metadata-only evidence, not proof of current
availability of the linked media or of rights in underlying recordings beyond
the returned item-level CC0 declarations. The manifest was not used to fetch
any media. Source policy links: [iNaturalist API](https://www.inaturalist.org/api),
[API recommended practices](https://www.inaturalist.org/pages/api+recommended+practices),
[photo and sound licensing](https://help.inaturalist.org/en/support/solutions/articles/151000175695),
and [video upload support](https://inaturalist.freshdesk.com/en/support/solutions/articles/151000169940-can-i-upload-videos-to-inaturalist-).

## Gate outcome and next step

This candidate fails the predeclared minimum-size gate and does not justify
media acquisition, feature extraction, fitting, or validation scoring. Keep it
excluded. Continue the R2 source search only for a task with item-level rights,
at least 1,000 usable examples and 200 independent groups, a frozen split, and
a design with unimodal and input-perturbation controls. Do not inflate counts
by adding repeated observations from the same source groups.

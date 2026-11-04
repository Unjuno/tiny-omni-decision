# Action Verb Corpus source screen — 2026-11-05

## Disposition

**REVIEW_NOT_ADMITTED; no data downloaded or opened.** The official OFAI
resource page describes a plausible small synchronized audio/video/motion
corpus, but its license statement conflicts with the exact Zenodo record from
the same provider. Do not use it for product-directed fitting until the
publisher resolves the conflict in writing.

## Source evidence

- The OFAI resource page describes 46 recordings by 12 people, 390 instances
  of take, put, and push, with audio, video and hand/arm motion while the
  participant performs and describes an action. It states CC BY 4.0 and German
  as the language ([OFAI Action Verb Corpus](https://www.ofai.at/resources/avc)).
- The linked official Zenodo record, DOI
  [10.5281/zenodo.5140014](https://zenodo.org/records/5140014), labels the
  uploaded dataset CC BY-NC-SA 4.0 and explicitly limits use to non-commercial
  purposes. Its file table lists 590.7 MB total, including a 192.6 MB
  video archive, motion/position archives, and annotations.
- Because these are conflicting statements from the publisher's own official
  pages, the less restrictive webpage is not treated as sufficient permission
  for commercial/product model development.

## Task and split limits

The recordings could support narrow action recognition from synchronized
streams, but the publisher says the participant both performs **and describes**
the action. Audio may therefore reveal the action directly; this source alone
does not define a task where both audio and video are necessary. The 12 people
and 46 recordings also leave few independent person/session groups for a
strong generalization estimate. These are feasibility observations, not
measured model results.

## Decision

- Do not access or train on the AV/motion payload under the unresolved license.
- Do not count this as an admitted R2-M source or as evidence of fusion quality.
- Reconsider only if the publisher clarifies the applicable license and a
  person/session-disjoint task is frozen with matched unimodal controls.

No checkpoint, feature cache, split, or evaluation was created.

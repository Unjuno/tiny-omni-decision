# Paired-task readiness: annotation and representation audit

## Executed training-metadata audit

Read only49 already frozen STARSS23 **training** annotation members after
verifying the training manifest SHA256
`571c2b537a809b2f92c9289fd932c20e0169ca1c0e38f4dc0f4c6ad6ef2ac978`
and metadata archive SHA256
`2fa848545a520b4a9d490459de7f079aa49ae16daf0affc859c016f5146981d6`.
No calibration/validation/evaluation annotation members or media were read by
this audit. Archive hashing reads opaque bytes, not held-out member contents.

Observed116,750 annotation rows,360 distinct azimuth values,119 distinct
elevation values, and distances28..470cm, with zero rows at distance0. Six fields
are tick (100ms), raw class, source index, azimuth, elevation and distance.
The schema contains no visibility or visual bounding-box annotation. Absence
of such a field does not prove visibility cannot be established by another
procedure, but prevents treating visibility as an existing ground-truth label.

Raw class annotation-row counts (not independent example counts):

| Class index | Rows |
|---:|---:|
| 0 | 35280 |
| 1 | 32704 |
| 2 | 151 |
| 3 | 664 |
| 4 | 4237 |
| 5 | 12015 |
| 6 | 2435 |
| 7 | 855 |
| 8 | 23240 |
| 9 | 4250 |
| 10 | 861 |
| 11 | 0 |
| 12 | 58 |

Source indices are not universal identity labels: the pinned README assigns
actor identities within a scene and uses0 for independent non-actor sources.
Do not manufacture cross-recording identity supervision from these integers.

## Actual existing feature path

`extract_starss23_features.py` uses the pinned **DyMN width0.4** model, checkpoint
`49b0e63577021994257b3de2ac5f1686d71554b93c8149f9d2568ad5760b904e`,
on audio channel0 only. This STARSS23 path is not the separate Whisper speech
path. The inspected paired manifest identifies FOA audio. The pinned source
README gives channel0's ideal response as direction-independent and the other
three channels as directional components. Therefore this mono event embedding
does not retain the explicit multichannel directional observations. Do not
claim all possible indirect room/direction cues vanish, but do not call it a
spatial audio representation.

The existing VideoPrism path averages every spatiotemporal token into768
numbers. The event-label parser validates but then discards azimuth/elevation/
distance. These concrete choices match the old event-presence task; they do
not support a new spatial audiovisual task without changing the representation
and defining new labels. Merely changing the fusion head will not restore the
discarded channels or token grid.

## New source checks, not admissions

On2026-10-12, consulted primary project pages:

- [AVQA](https://mn.cs.tsinghua.edu.cn/avqa/) has manual four-option questions
  tagged View/Sound/Both, so its task form is relevant. Raw clips come from
  VGGSound/YouTube. Its displayed license text describes restricted ACM copying,
  not an unambiguous permissive grant for all underlying media. **REVIEW**, no
  ingestion; a third-party CC-BY mirror tag is insufficient to resolve rights.
- [MUSIC-AVQA](https://github.com/GeWu-Lab/MUSIC-AVQA) is relevant to audiovisual
  reasoning, but the page shows a license-file MIT badge while its README says
  GPLv3. Neither observation establishes permission for all underlying videos.
  **REVIEW**, no code copied, no media/annotations downloaded or trained on.

These are scoped readiness findings, not legal determinations or a conclusion
that no usable audiovisual dataset exists.

## Decision and next executable step

Do not relabel existing STARSS23 event presence as proof that a question requires
both modalities. Do not use different timestamps/recordings as semantic negative
labels when repeated actions may still match. Preserve all previous scores.

The already approved STARSS23 recordings provide multichannel spatial data and
time-indexed source locations. A bounded **training-only spatial feature check**
can first establish channel orientation, units and annotation agreement before
building a spatial readout. Vision should retain its token grid if a subsequent
task genuinely benefits from visual location. Audio-only localization would
still be a control, not proof of mandatory AV fusion. A task requiring visible
source association needs additional justified supervision, not inferred boxes
or visibility labels silently promoted to truth. The R2-M complementary AV/VS/
AS/AVS task matrix remains open.

Reproduce metadata audit: `python -m scripts.audit_starss23_spatial_training`.
Artifact:`C:/CodexArtifacts/pretrained-reuse-r2/starss23-spatial-training-audit-01/result.json`.
It contains per-recording counts, all pins and exact executed script hash;
subsequent source edit only wrapped a validation line. No learning, media
acquisition, paid compute, sealed audit or legacy artifact modifications.

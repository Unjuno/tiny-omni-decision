# AVSET-10M metadata-only R2 gate — 2026-11-15

## Disposition

Keep AVSET-10M `REVIEW_NOT_ADMITTED` for product-directed Path A / M-2
evaluation. It is a plausible audio-video correspondence source, but the
available metadata and rights do not establish an eligible, independently
grouped decision benchmark. Do not acquire clips or start feature/model work
for this candidate.

## Pinned evidence

- Author repository: `AVSET-10M/avset-10m`
- Inspected revision: `d9b3db810ab273328eb8c1819cb39bd8561845fd`
- The author README describes AVSET-700K as 727,530 AudioSet-derived clips
  with manually labeled audio categories and AVSET-10M (excluding that subset)
  as 9,877,475 Panda-70M-derived clips with pseudo-labels. Metadata include
  YouTube ID, segment times, an audio-visual correspondence score, and labels.
- This supports a possible candidate-video selection task, but AVSET does not
  provide a frozen supplied-option Decision benchmark. A new task would need
  group-safe negative construction and controls to show that both audio and
  video are necessary.
- The author repository's CC BY 4.0 license applies to the repository Work;
  its README separately directs users to follow the source-video license. The
  linked HD-VILA-100M agreement permits only non-commercial research and
  explicitly excludes using Data or Results in a product or service. The
  clips are drawn from YouTube-backed AudioSet and Panda-70M sources. No
  per-video product-use permission roster was established, so the dataset
  repository license alone does not clear the media payload for this use.
- No canonical group-disjoint split or unique parent-video/group counts were
  established. Thus the provisional requirement of at least 1,000 examples
  and 200 independent groups is unverified. Prior exposure of the relevant
  parent videos is also unknown.
- The 10M pseudo-labels are not a basis for claiming human-verified Decision
  targets. The 700K subset's audio category labels do not themselves establish
  accurate cross-modal option labels.

## Work boundary

This was a pinned repository README/LICENSE and official project-page review
only. No dataset shards, YouTube media, annotation payloads, model checkpoints,
features, predictions, GPU or evaluation split were accessed. No existing
benchmark or artifact was changed.

## Sources

- [Pinned author README](https://github.com/AVSET-10M/avset-10m/blob/d9b3db810ab273328eb8c1819cb39bd8561845fd/README.md)
- [Pinned author repository license](https://github.com/AVSET-10M/avset-10m/blob/d9b3db810ab273328eb8c1819cb39bd8561845fd/LICENSE)
- [Source-video terms linked by AVSET](https://github.com/microsoft/XPretrain/blob/main/hd-vila-100m/LICENSE)
- [Author project page](https://avset-10m.github.io/)
- [AudioSet publisher download and license page](https://research.google.com/audioset/download.html)

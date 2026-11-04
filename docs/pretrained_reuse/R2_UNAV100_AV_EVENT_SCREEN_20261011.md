# Joint audio-video source screen: UnAV-100 (2026-10-11)

## Decision

Add UnAV-100 as a review-only candidate for synchronized audio-video event
decisions. The official project describes over 10,000 untrimmed videos, over
30,000 audiovisual events across 100 categories, and temporal event boundaries.
Its raw-video index is described as YouTube ID, start/end seconds, and official
train/validation/test split. These properties fit a bounded event-presence or
temporal-localization Decision task better than subjective AV quality scores.

The official page says the dataset is CC BY 4.0 for commercial and research use,
but also explicitly says copyright remains with the original video owners.
Therefore the annotation/feature release is promising, while the original media
remain `PER-SOURCE`/`REVIEW` until source-level rights and identity are checked.
The Hugging Face mirror currently advertises CC BY-NC-SA 4.0, which conflicts
with the official project-page statement. That mirror is explicitly excluded;
do not silently apply the project-page grant to mirror artifacts. No video,
audio, annotation, or feature archive was downloaded. The code repository's
pinned commit is not treated as a data revision or archive hash. The candidate
is not included in a training or evaluation catalog.

## Provenance and limits

- Official project: `https://unav100.github.io/`.
- Official code repository commit: `ttgeng233/UnAV@1609deba4fbaf25222c6ee6c7d43b0d0ddfab20d`.
- Official project page describes >10K videos, >30K events, 100 categories,
  temporal boundaries, and a YouTube-ID/timestamp/split CSV.
- The same page licenses the dataset under CC BY 4.0 and reserves underlying
  video copyright to original owners.
- The Hugging Face mirror page advertises CC BY-NC-SA 4.0. Its license scope
  and relationship to the official release were not reconciled; no mirror
  artifact is approved or used.
- Annotation and feature download is linked through Baidu Drive; availability,
  exact files, hashes, and feature encoders have not been verified.
- No row-level source identity, deduplication, rights audit, media decoding, or
  model evaluation was performed.
- Keep `manifests/candidates/unav100.yaml` at `trust_status: review`; do not
  change allowlists or treat a dataset-level license statement as per-video
  authorization.

## Sources

- [Official UnAV-100 project and license statement](https://unav100.github.io/)
- [Official UnAV code repository](https://github.com/ttgeng233/UnAV)
- [Hugging Face mirror card advertising CC BY-NC-SA 4.0](https://huggingface.co/datasets/ttgeng233/UnAV-100)
- [Creative Commons Attribution 4.0 deed](https://creativecommons.org/licenses/by/4.0/)

This screen identifies a promising task source; it does not clear data use or
close the synchronized-AV gate. It does not establish that any source-owner
license, split, benchmark, or pretrained feature is usable in the product.

# AVMIT audiovisual data quality and rights screen — 2026-11-26

## Disposition

**Promising data design; `REVIEW_NOT_ADMITTED` for this product-directed path.**
AVMIT is a substantially stronger audio/video data candidate than incidental
same-observation media pairs: trained raters evaluated each three-second clip
with both audio and video, three ratings are retained per clip, and the task
distinguishes an absent event, a present-but-not-prominent event, and a
prominent event. The publisher reports 57,177 annotated clips and a separate
960-clip curated test set. No annotation CSV, feature archive, raw video,
evaluation label, or model output was downloaded or opened in this screen.

The core caveat is provenance. AVMIT's original videos come from the Moments
in Time (MiT) dataset. AVMIT's GitHub repository labels its code and CSV
repository CC0, and Zenodo record 8253350 labels its files CC BY 4.0. Those
grants do not establish commercial/derivative-training rights for the
underlying MiT videos or necessarily for representations computed from them.
AVMIT's paper says raw MiT access requires an application; the official MiT
site was unavailable during this check. Secondary published sources describe
MiT as restricted to non-commercial research/education. Until the MiT rights
and the feature archive's scope are clarified by the upstream custodians, do
not ingest the embeddings or train on them.

## Pinned metadata inspected

- AVMIT Git repository: `mjoannou/audiovisual-moments-in-time`, observed
  `main` commit `e3c06ec0e215b1934d141f86008e92e567366d6c` on 2026-11-26.
  The repository page advertises a CC0-1.0 license and lists the rating CSVs,
  test-set CSV, and feature-reader code. This repository license is not proof
  of rights to the underlying videos or separately hosted feature archive.
- Zenodo record: version `10.5281/zenodo.8253350`, record revision 2, published
  2023-08-16; metadata reports `cc-by`. File metadata only:
  `video_ratings.csv` 6,867,402 bytes (MD5
  `9817e7adaaa8aa7d2384823b9d618e7e`);
  `test_set.csv` 121,081 bytes (MD5 `b78453da858755f889cadcf9b0300e22`);
  `AVMIT_VGGish_VGG16.tar` 498,708,480 bytes (MD5
  `11c523793f702c98c41fb5b30a6d7fee`);
  `AVMIT_YamNet_EffNetB0.tar` 2,063,124,480 bytes (MD5
  `45c51034fb418eb0bbfdae2d17afb29b`). These are metadata checksums, not
  locally verified file hashes.
- The AVMIT article is CC BY 4.0. It reports 57,177 clips annotated by three
  trained participants each, 23,160 majority-confirmed audiovisual events,
  and 17,891 clips where the event is majority-rated prominent. The curated
  test collection has 960 videos over 16 classes. These counts are publisher
  reports, not this project's audit.
- The article describes 41 selected audiovisual classes and says MiT videos
  were screened to include non-silent audio. Its authors also explicitly
  discuss that audiovisual labels in other corpora may not indicate the same
  underlying event. AVMIT's rater procedure is a meaningful data-quality
  advantage, though it does not guarantee every class requires both streams.
- Proposed task adaptation, if rights clear: score the supplied choices
  `absent`, `present but not prominent`, and `prominent` for the named AVMIT
  event, using the three-rater distribution as a soft target. Compare audio
  only, video only, joint, option/prior-only, shuffled, missing, and
  cross-example audio/video controls. Do not treat the paper's curated 960
  videos as iterative validation; reserve them until a frozen candidate and
  protocol are ready, and first verify whether they are already exposed by
  prior AVMIT research.

## Data-quality gates before reconsideration

1. Obtain explicit permission or authoritative license terms for the MiT
   source videos and for model training from the published AVMIT embeddings.
   Do not submit access forms or personal data on the user's behalf.
2. Confirm the annotation and embedding files do not include the separate
   curated test videos; identify them by opaque clip ID without opening their
   labels or evaluating them.
3. Derive train/validation splits from source-video identity, not class-folder
   path. Check duplicate source IDs and near-duplicate clips; group all clips
   from one source video together. Check any existing project video identity
   ledgers before admission.
4. Audit class counts, rater agreement, soft-target entropy, and video-level
   label disagreement on training rows. Preserve all three rater outcomes;
   do not discard hard or ambiguous clips to inflate accuracy.
5. Predefine a development split and the modality-dependence controls before
   extracting features or fitting a readout. Keep the curated test set sealed.
6. Establish that the published feature vectors are adequate for this goal:
   they come from VGGish/VGG-16 or YamNet/EfficientNet-B0, not the project's
   audited VideoPrism/EfficientAT path. A result on them would be a data/task
   feasibility study, not quality evidence for the target encoder stack.

## Current decision

AVMIT ranks high on annotation quality and genuine audiovisual alignment, but
does not currently pass the source-rights gate, and its precomputed features
do not match the target encoders. Keep it as a leading candidate for a future
publisher clarification, not as admitted training data. This screen does not
close R2 M-2, justify a model change, or support any product-quality claim.

## Sources

- [AVMIT article and data description](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0301098)
- [Pinned AVMIT Git repository](https://github.com/mjoannou/audiovisual-moments-in-time/tree/e3c06ec0e215b1934d141f86008e92e567366d6c)
- [Zenodo version 10.5281/zenodo.8253350](https://zenodo.org/records/8253350)
- [MiT official dataset page](https://moments.csail.mit.edu/)
- [MiT access instructions in OpenMMLab's loader documentation](https://github.com/open-mmlab/mmaction2/blob/main/tools/data/mit/README.md)
- [Published report describing MiT's non-commercial research/education restriction](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136760487-supp.pdf)

## Reproducibility note

This was a read-only source screen plus metadata-only Zenodo API inspection.
The Zenodo record JSON was not saved in the repository. The source revision,
record ID/revision, sizes, metadata checksums, and URLs above are the pinned
receipt needed to repeat the metadata check. No tests are relevant to this
documentation-only change.

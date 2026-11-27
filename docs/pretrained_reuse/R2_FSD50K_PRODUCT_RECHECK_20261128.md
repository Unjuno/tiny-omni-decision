# FSD50K product-use and BSD10k label-fit recheck — 2026-11-28

## Decision

Keep FSD50K on hold for product-directed training. Its eligible CC0/CC-BY
positive examples may be useful for a separately defined research task, but the
currently documented evidence does not clear the product-use rights question
or support adding those examples as negatives/labels in BSD10k's 23-way task.
No audio or evaluation split was fetched or opened in this recheck, and no
maintainer was contacted.

## Primary-source findings

The official [FSD50K Zenodo record](https://zenodo.org/records/4060432) states:

- Development labels are clip-level weak labels; labels are described as
  correct but occasionally incomplete. A missing label therefore cannot be
  treated as proof that a sound is absent.
- The eval set is separately described as exhaustively labeled. It remains
  outside this recheck and must not be used for training or tuning.
- Clip licenses are mixed. The record says per-clip licenses are specified in
  `dev_clips_info_FSD50K.json` and `eval_clips_info_FSD50K.json`; the dataset as
  an entity also has CC-BY, while the record cautions that the scope of a
  global dataset license is not straightforward.
- For commercial use, the record directs users to contact the maintainers.

The existing [2026-10-13 feasibility screen](R2_FSD50K_ADMISSION_20261013.md)
recorded 31,566 eligible train clips after excluding per-clip NC and
Sampling+ licenses, plus counts for eight candidate positive labels. That is
metadata coverage only, not proof of rights for a product use case, media
integrity, or a supervised 23-class crosswalk.

## Fit to BSD10k

BSD10k uses 23 second-level classes across sound effects, instrument samples,
music, soundscapes, and speech. FSD50K is a 200-class multi-label AudioSet
ontology dataset with upward-propagated labels; dev annotations may be
incomplete. The existing eight-label presence counts do not define a unique
mapping into BSD10k's 23 mutually exclusive labels. Assigning unlisted FSD50K
labels as negatives would create unsupported negatives, and collapsing
ambiguous positives into BSD categories would create noisy targets.

Before reconsidering FSD50K, both conditions must be met:

1. Rights for the intended product use are clarified against the official
   commercial-use contact instruction and per-clip licenses.
2. A frozen, human-reviewable mapping identifies only defensible training
   positives, preserves unknown labels as unknown, groups by uploader, and
   passes cross-source duplicate checks. It must not alter the existing BSD10k
   validation/test labels or selection protocol.

Until then, keep the existing BSD10k result and FSD50K positive counts as
separate evidence. Do not use the unopened BSD10k test, FSD50K eval split, or
sealed audit to tune a mapping or model.

## Local artifact check

The current C-drive FSD50K artifact directory is absent. The historical
`C:\CodexArtifacts\pretrained-reuse-r2` path is a junction to
`D:\CodexArchiveRelocated\CodexArtifacts\pretrained-reuse-r2`; the target
drive is unavailable in the current environment. No junction, archive, or
legacy artifact was modified or repaired. This recheck used the official
Zenodo page and repository documentation only.

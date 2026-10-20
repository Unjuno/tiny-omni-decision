# BatVision audio–visual source screen — 2026-10-21

## Decision

**REVIEW / not admitted to this run.** BatVision is a strong task-shaped
candidate for synchronized audio, image/video, and depth-sensor decisions, but
its public archive is far larger than available local storage. Do not begin a
partial download, extract, feature build, or model run from the current archive.
No media, labels, split CSV contents, or test examples were opened.

## Primary-source evidence

- Official dataset repository: `AmandineBtto/Batvision-Dataset`, pinned Git
  commit `306eb09ca5fdbd7c527a600822a23ce29ebac1d3`. The metadata-only clone
  contains 18 files / 67,590 bytes; it has no split CSVs or media.
- The official README states that binaural echoes are synchronized with
  corresponding RGB-D images and provides train/validation/test CSV lists for
  both sites. The paper reports 52,220 UC Berkeley (BV1) and 3,120 Mines Paris
  (BV2) observations. BV1 test is from another floor; BV2 uses time-separated
  trajectory segments, but exact room/session group independence still needs
  row-level verification before a claim of independent validation.
- The official repository says the dataset is **CC BY-SA 4.0**. Preserve
  attribution and share-alike implications for any redistributed adaptation;
  whether learned weights count as an adaptation is unresolved here. The source
  was not added to an approved product-training manifest.
- The official Dataverse API metadata for DOI `10.57745/HYLZNL` lists five
  archive shards totaling **168,536,422,535 bytes**. This is metadata only; no
  shard was downloaded. C: had **28,028,788,736 bytes free** at the check.
- GPU use was not attempted. At the check, the RTX 3080 was occupied by the
  Orbitune process (PID 23536); no process was signalled or modified.

## Task fit and limits

The data could support typed decisions such as choosing among depth bands or
traversability options from an audio echo, a camera observation, or both, with
depth maps as labels. It is a promising natural sensor task aligned with the
project's temporal/physical-decision direction. However, the paper also
demonstrates audio-only depth prediction, so the source alone does **not** prove
that a chosen label requires both modalities. Any fusion benefit would need
matched audio-only, vision-only, and joint controls. A joint-required M-2 task
cannot be claimed from paired timestamps alone.

The paper describes individual synchronized chirp/image observations rather
than a ready-made decision sequence API. A temporal probe would need to recover
route sequences from capture metadata and split by route/room/site before
constructing question windows. Those identities and media-content overlap have
not been audited. The released depth target is pixelwise, so any conversion to a
small set of typed options must be preregistered and tested against invalid or
missing depth regions rather than chosen after viewing model scores.

## Reproduction trail and references

- Official dataset/code repository and license: <https://github.com/AmandineBtto/Batvision-Dataset>
- Official project page and download link: <https://amandinebtto.github.io/Batvision-Dataset/>
- Dataset paper: <https://arxiv.org/abs/2303.07257>
- Dataverse record / API metadata: <https://entrepot.recherche.data.gouv.fr/dataset.xhtml?persistentId=doi:10.57745/HYLZNL> / `https://entrepot.recherche.data.gouv.fr/api/datasets/:persistentId?persistentId=doi:10.57745/HYLZNL`
- Metadata-only local copy: `C:\CodexArtifacts\public-source-screens\batvision-official\`

## Next gate

Reconsider only if a rights-compatible, independently auditable subset can be
obtained without the 168.5 GB archive, or if storage is intentionally expanded
without deleting existing artifacts. Before any training, verify the exact
per-file license scope, split/room/route IDs, media hashes, audio/image
synchronization, depth-map validity, and a fixed typed-decision conversion. If
those checks pass, use a separate experiment/output namespace and the
cooperative HotUGPU wrapper. This screen does not change the four-modality
quality target, parameter budget, or product status.

# V-JEPA 2.1 ViT-B checkpoint rights and artifact recheck — 2026-11-14

## Decision

The checkpoint remains `REVIEW_REQUIRED` for product-directed use. Its identity,
local bytes, and an earlier strict runtime load are verified. The current
official materials do not explicitly grant this specific checkpoint a license;
repository source-code terms are not treated as a checkpoint-specific grant.
No change to the admitted Path A encoder set follows from this recheck.

## Pinned official sources

- Official repository revision: <https://github.com/facebookresearch/vjepa2/tree/204698b45b3712590f06245fbfba32d3be539812>
- Pinned [README](https://github.com/facebookresearch/vjepa2/blob/204698b45b3712590f06245fbfba32d3be539812/README.md) and [LICENSE](https://github.com/facebookresearch/vjepa2/blob/204698b45b3712590f06245fbfba32d3be539812/LICENSE).
- `main` observed on 2026-11-14: `204698b45b3712590f06245fbfba32d3be539812`.
- Official README lists the V-JEPA 2.1 ViT-B/16 checkpoint (rounded 80M
  parameters) and its download link. The README's License section says the
  majority of V-JEPA 2 is MIT-licensed and separately identifies three source
  files under Apache-2.0; it does not expressly state that the MIT license
  covers this external checkpoint file.
- Meta's project page describes release for community use but does not state a
  checkpoint license or usage/redistribution terms:
  <https://ai.meta.com/research/vjepa/>.

## Official object and local artifact

A read-only `HEAD` to the official URL returned HTTP 200 with:

- URL: `https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt`
- Content-Length: `1,664,223,428` bytes
- S3 version ID: `xJBU4AkoA4gv5boeC6gOA7eElzWcubxY`
- Last-Modified: `2026-03-16T09:12:19Z`
- ETag: `be0dc26f052ae6a7476714cd53176836-199` (multipart ETag, not SHA-256)

The existing local artifact at
`C:\CodexArtifacts\pretrained-reuse-r1\vjepa2_1_vitb_dist_vitG_384.pt`
was rehashed without loading it:

- bytes: `1,664,223,428`
- SHA-256: `848a77c33cc9e6649ed2119c9bea1e2c569bcdab9539ff3e7c02ccc2959ddf4d`
- last write: `2026-10-10 23:07:00`

The byte count and SHA-256 match the retained audit receipt. The source
repository revision and S3 version are unchanged from the prior checkpoint
recheck.

## Runtime evidence and limits

The retained read-only runtime receipt
`C:\CodexArtifacts\pretrained-reuse-r1\vjepa21-audit.json` records the
previous official-factory load at source revision `204698b45b3712590f06245fbfba32d3be539812`:

- strict checkpoint loading; FP32 encoder; 86,833,152 runtime encoder parameters
- finite image and 8-frame video outputs
- CUDA peak allocated memory: `831,897,600` bytes
- smoke inputs were plumbing fixtures, not task-quality samples

No model was reloaded or forwarded in this recheck; the GPU was occupied by a
separate Orbitune process. No checkpoint, model cache, or run was modified.
The existing runtime receipt proves load/forward compatibility only, not
quality, a valid license, or inclusion in the <=200M complete product budget.

## R1 disposition

Keep this checkpoint out of product-directed fitting and packaging unless a
checkpoint-specific grant or written permission applicable to the intended use
is established. The already audited VideoPrism-B remains the Path A video
feature source; this recheck does not change its status.

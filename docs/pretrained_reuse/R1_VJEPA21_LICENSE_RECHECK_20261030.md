# V-JEPA 2.1 ViT-B checkpoint rights recheck (2026-10-30)

## Decision

The exact checkpoint remains **REVIEW** for product use. This recheck confirms
that the local file still matches the previously audited SHA-256 and that the
official distribution object is stable and versioned, but the current official
repository and Meta project page still do not provide a checkpoint-specific
license grant for `vjepa2_1_vitb_dist_vitG_384.pt`.

## Current official-source inspection

- `git ls-remote https://github.com/facebookresearch/vjepa2.git refs/heads/main`
  returned `204698b45b3712590f06245fbfba32d3be539812`, the same source commit
  already pinned in the R1 audit.
- The [official README at that commit](https://github.com/facebookresearch/vjepa2/blob/204698b45b3712590f06245fbfba32d3be539812/README.md)
  says the majority of V-JEPA 2 is MIT-licensed and lists three source files
  separately under Apache-2.0. It does not name the external ViT-B checkpoint
  or expressly extend those code terms to that file.
- The [repository LICENSE](https://github.com/facebookresearch/vjepa2/blob/204698b45b3712590f06245fbfba32d3be539812/LICENSE)
  governs the repository software; it does not identify this S3 checkpoint.
- [Meta's V-JEPA project page](https://ai.meta.com/research/vjepa/) says it is
  releasing V-JEPA 2 for the community to build upon, but does not specify
  license, redistribution, or derivative-weight terms for this V-JEPA 2.1
  checkpoint.

## Checkpoint object and local artifact

A read-only HTTP `HEAD` request to the official distribution URL returned:

- URL: `https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt`
- Content-Length: `1,664,223,428` bytes
- ETag: `be0dc26f052ae6a7476714cd53176836-199` (multipart ETag; not treated as
  a SHA-256)
- S3 version ID: `xJBU4AkoA4gv5boeC6gOA7eElzWcubxY`
- Last-Modified: `2026-03-16T09:12:19Z`

The existing local file at
`C:\CodexArtifacts\pretrained-reuse-r1\vjepa2_1_vitb_dist_vitG_384.pt`
is `1,664,223,428` bytes with SHA-256
`848a77c33cc9e6649ed2119c9bea1e2c569bcdab9539ff3e7c02ccc2959ddf4d`, matching
the earlier frozen audit receipt. Its previously saved runtime receipt,
`C:\CodexArtifacts\pretrained-reuse-r1\vjepa21-audit.json`, records strict
EMA loading, finite image and 8-frame outputs, 86,833,152 encoder parameters,
and a 831,897,600-byte peak CUDA allocation. That receipt used plumbing inputs;
it is not task-quality evidence. No model was loaded or run in this recheck.

## Scope and next gate

No bytes were downloaded, no code/model artifact was modified, and no third
party was contacted. The stable S3 version ID makes the object more precisely
identifiable but does not grant rights. Keep this encoder research-only until
the exact checkpoint has an explicit applicable license statement or written
permission covering intended use. The separately audited VideoPrism-B remains
the currently admitted video feature source in Path A; this decision does not
change its status.

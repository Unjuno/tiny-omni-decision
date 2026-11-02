# VideoPrism–AVATAR source overlap audit (2026-11-02)

## Finding

Exact overlap between the pinned VideoPrism-B checkpoint's pretraining videos
and the 2,268 YouTube source groups referenced by the AVATAR TRAIN inputs is
**unknown**. The reviewed official VideoPrism materials describe pretraining at
large scale (including YT-Temporal-180M, InternVid, VideoCC and WTS-70M), but do
not publish a checkpoint-specific item-level video-ID manifest. The local
pinned source checkout has no such manifest either. This establishes neither
zero overlap nor a known positive match. Since AVATAR groups are YouTube IDs and
some named pretraining sources are YouTube collections, possible source overlap
is material; this is an inference, not an observed collision.

The AVATAR held-category result therefore remains a **source-group-disjoint,
category-held-out development diagnostic within AVATAR TRAIN**. It must not be
described as independent-source transfer evidence. Its category holdout and
source-group separation still describe the AVATAR split protocol; they do not
establish independence from VideoPrism pretraining.

## Pinned inputs

- VideoPrism checkpoint: `google/videoprism-base-f16r288`, revision
  `be719a406d563b66f0ac969e7c94bab8e997c81a`; weights SHA-256
  `61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29`.
- Official implementation revision:
  `c6eba0ece3dd49a24ffc44582966997a4a59f1d8`.
- AVATAR dataset revision:
  `a7f0146ff704799d044133e61137836cacc1c233`; TRAIN model input SHA-256
  `0139f4f59a685c3f83890156b59c42d12aca9db54b22ee5c0a9c70b6ec7d60ef`;
  sidecar SHA-256
  `90adbf04450e0854fdb379dbcceb6aaa73790b205c08f1cf6c2cb786b6710a57`.
- The pinned AVATAR sidecar references 2,268 unique source groups. SHA-256 of
  sorted group IDs joined with LF:
  `b0075660b35ca5538e5afd878a455b6b78897daed0b0b637c88162b81d6e1f55`.

## Scope and handling

This was a read-only provenance check. It did not load the model, extract
features, recompute metrics, download data, inspect official validation labels,
or access sealed audit data. No individual source IDs are reproduced here. The
machine-readable record is
[`videoprism-avatar-overlap-audit-20261102.json`](../../manifests/pretrained_reuse/videoprism-avatar-overlap-audit-20261102.json).

Sources: [Google Research's VideoPrism pretraining description](https://research.google/blog/videoprism-a-foundational-visual-encoder-for-video-understanding/)
and the [pinned official VideoPrism README](https://github.com/google-deepmind/videoprism/blob/c6eba0ece3dd49a24ffc44582966997a4a59f1d8/README.md).

# ARID video task/source screen — 2026-11-04

## Disposition

**`REVIEW_NOT_ADMITTED`; no data acquisition or evaluation.** ARID v1.5 could
support a narrow video-only action decision in low-light conditions, but the
available source records conflict on dataset access/licensing, and the public
split evidence does not establish a fresh person/scene-disjoint split for v1.5.
This is not an R2-M both-input task and cannot establish cross-modal benefit.

## Evidence checked (metadata and documentation only)

- The [official ARID project page](https://xuyu0010.github.io/arid.html) describes
  v1.5 as 5,572 clips, 11 actions, 24 scenes and more than 15 volunteers, and
  advertises a CC BY 4.0 license. The same page offers a Drive archive. The
  wording says “this work” is CC BY 4.0 without a version-specific grant or
  separate notice for underlying participant/personality rights.
- The authors' [baseline repository](https://github.com/xuyu0010/ARID_v1)
  contains code and a CC-BY-4.0 license, but its README says to contact the
  author for the dataset download. The repository is the v1 baseline; its
  license file alone does not establish access to v1.5 media.
- The [original ARID paper](https://arxiv.org/html/2006.03876) describes v1.0:
  3,784 clips, 11 volunteers and 18 scenes. It divides each action class into
  groups based on similar lighting or the same actor, then makes three random
  70/30 group splits. This does not establish that a person or scene is held
  out across *all* action classes, and it does not provide an auditable v1.5
  person/scene split manifest.
- A [2026 ARID v1.5 study](https://link.springer.com/article/10.1007/s44163-026-02020-6)
  says the v1.5 data was used under license and is not publicly available; it
  says access is by reasonable request and with permission from data providers.
  This conflicts with interpreting the project page's CC-BY wording as an
  unconditional grant for the v1.5 media.

## Gate assessment

| Gate | Finding |
|---|---|
| Task relevance | Plausible narrow video M1 task: select the action class from low-light video. Not joint multimodal evidence. |
| Dataset-specific rights | **Unresolved.** The public page, baseline repo instructions and later v1.5 access statement do not agree on whether media can be obtained and used without provider permission. |
| Independent groups | **Not verified for v1.5.** Original v1 groups do not prove cross-action person/scene isolation; no accessible v1.5 group manifest was found. |
| Model pretraining overlap | **UNKNOWN.** No source-level inventory proves the candidate encoder excludes ARID. |
| Resource feasibility | **UNKNOWN.** No media sizes or checksums were read; no download attempted. |
| Decision | Keep `REVIEW_NOT_ADMITTED`; do not train, evaluate, or select a checkpoint on ARID. |

## Evidence boundary and next gate

No dataset payload, Google Drive archive, model weight, split label, or sealed
audit was accessed. No message was sent to the provider. Reconsider only after a
version-specific written license/access basis is available, an immutable
v1.5 file/hash inventory can be built, and split metadata supports a
cross-action person/scene-disjoint development design. If these conditions
pass, preregister a single frozen-feature Video M1 probe and retain separate
train, calibration, and development groups. Do not treat the dataset's
published benchmark scores as Path A results.

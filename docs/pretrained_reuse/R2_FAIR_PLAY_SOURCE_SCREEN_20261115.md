# FAIR-Play source and task screen — 2026-11-15

## Disposition

**`REVIEW_NOT_ADMITTED` for a quality run.** FAIR-Play is a promising paired
audio/video source for a narrow cross-modal correspondence diagnostic, but the
available source documentation does not establish participant-rights scope or
a sufficiently large independent-group split. No media, split payloads,
features, model outputs, or sealed evaluation data were accessed. No Git LFS
objects or repository clone were fetched.

## Pinned source evidence

- The archived [official repository at commit
  `2666d6caf2d126ae43a84841e94e0c9ae24b3e2d`](https://github.com/facebookresearch/FAIR-Play/tree/2666d6caf2d126ae43a84841e94e0c9ae24b3e2d)
  states that FAIR-Play is licensed CC BY 4.0 and contains 1,871 roughly
  aligned video/binaural-audio clips from a music room. Its README says the
  repository is about 100 GB and offers ten random train/validation/test
  splits. Those random splits are not evidence of participant-disjointness.
- The [pinned license file](https://github.com/facebookresearch/FAIR-Play/blob/2666d6caf2d126ae43a84841e94e0c9ae24b3e2d/LICENSE)
  is CC BY 4.0. The repository tree has audio/video and split files stored
  through Git LFS; no LFS objects were fetched.
- The [authors' CVPR 2019 paper](https://www.cs.utexas.edu/~grauman/papers/CVPR19_2.5d-visual-sound.pdf)
  says the clips are ten seconds each, totaling 5.2 hours, collected in one
  approximately 1,000-square-foot music room, with 20 volunteers playing solo,
  duet, and multiplayer performances. It describes ten random splits of
  1,497/187/187 clips. The paper does not document participant consent,
  likeness/voice release scope, or a participant-disjoint split. The README
  points to five harder scene-overlap splits in a separate repository, but
  those were not inspected here.

## Potential Decision task and gates

A task could ask whether a supplied audio/video pair comes from the same
recording, with hard mismatches chosen from similar instrument/performance
groups. That would require both streams by construction. This is a proposed
task adaptation, not an upstream label set or an admitted evaluation. A valid
protocol would still need matched audio-only/video-only, shuffled-pair,
missing-input, and candidate-order controls, with grouping by performer and
performance session.

| Gate | Finding | Disposition |
|---|---|---|
| Dataset license | Official repo applies CC BY 4.0 | Promising; authority and participant releases still need confirmation |
| Human/recording rights | 20 volunteers are visible/audible; paper does not state consent or downstream likeness/voice scope | Unresolved; fail closed pending source-owner evidence |
| Independent groups | 20 volunteers and one recording room are documented; random clip splits may repeat performers and scene | Does not establish the provisional 200 independent-group floor |
| Supplied-option task | No native Decision labels; correspondence labels would be derived from aligned file pairs | Proposed adaptation only |
| Both-input necessity | Pair correspondence could require comparing both streams, but it has not been tested | Not established |
| Exposure | No media or model output was accessed in this screen | Metadata-only; not a clean-source/exposure certificate |

Do not download the approximately 100-GB repository or fit on its random
splits. Reconsider a small feasibility diagnostic only if the source owner
confirms applicable participant/recording rights and a performer/session split
can be frozen with clear train/evaluation roles. Any result below the
independent-group floor remains diagnostic and cannot pass the product task
gate. This source does not establish general audio/video Decision quality.

# FAIR-Play license and group follow-up — 2026-10-22

## Decision

The dataset-level license declaration is now verified against the exact pinned
upstream release. FAIR-Play remains a **research candidate, not admitted for
quality-selected fitting**: source-scene IDs are absent from the split lists,
the usable task and hard-negative protocol are not frozen, and a paired label
does not by itself prove that both modalities are necessary.

## Pinned source evidence

- Repository: `facebookresearch/FAIR-Play`
- Revision: `2666d6caf2d126ae43a84841e94e0c9ae24b3e2d`
- The pinned [README](https://github.com/facebookresearch/FAIR-Play/blob/2666d6caf2d126ae43a84841e94e0c9ae24b3e2d/README.md)
  identifies the repository as the FAIR-Play dataset, declares CC BY 4.0,
  states that same-index video/audio pairs are roughly aligned, and reports
  about 100 GB for 1,871 clips.
- The pinned [`LICENSE`](https://github.com/facebookresearch/FAIR-Play/blob/2666d6caf2d126ae43a84841e94e0c9ae24b3e2d/LICENSE)
  is the Creative Commons Attribution 4.0 International Public License. Its
  Git blob SHA is `a6d7fd364bcc113164fb923e4eb53ae6fd75cae4`; its path and blob
  were verified in the pinned Git tree. The license grants only rights that
  the licensor has authority to grant, so this records the upstream release
  terms without making a broader legal opinion.
- The pinned [CVPR paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Xu_Visually_Informed_Binaural_Audio_Generation_without_Binaural_Audios_CVPR_2021_paper.pdf)
  describes recovering long recordings, grouping clips by scene, and creating
  five scene-aware splits. It does not supply an explicit original-scene key
  with the split lists available in the repository.

## Local evidence and remaining controls

- The existing train-only preflight contains eight video/audio pairs from
  PseudoBinaural split 1 train. Their media hashes and decoder receipt are
  preserved under
  `C:/CodexArtifacts/pretrained-reuse-r2/fair-play-train-group-preflight-attempt-01/`.
  No new media was downloaded, decoded, or modified in this follow-up.
- At inspection before the audio check, C: had 0.22 GB free and the RTX 3080
  Laptop had 15,821 / 16,384 MiB allocated to external PID 30196. No GPU work
  or full-corpus acquisition was started.
- The five train/validation manifests have row-level separation per split.
  Their validation lists overlap and no machine-readable scene IDs are
  present. See the [corrected split audit](R2_FAIRPLAY_PSEUDOBINAURAL_SPLIT_AUDIT_20261022.md).
- Same-index pair identity can support a controlled matching hypothesis, but
  random mismatched pairs may be trivially separable and audio-only or
  video-only signals may solve a weak candidate task. A train-only task
  construction must specify same-class hard negatives and compare matched
  audio-only, video-only, and joint controls before a quality fit.

## Admission status

| Gate | Status | Evidence / remaining work |
|---|---|---|
| Upstream dataset license declaration | **Verified: CC BY 4.0** | Pinned README and LICENSE path/blob at the release revision |
| Exact audio/video asset pair relation | **Partial** | README says same-index clips are roughly aligned; eight train-only WAV/MP4 audio waveforms now show a detectable pairwise match, but no physical AV sync or original scene identity is proven |
| Train/validation row separation | **Verified** | Existing pinned manifests and corrected metadata audit |
| Original-scene group count | **Unknown** | No group key in the audited lists; author-reported construction only |
| Both-input-required Decision task | **Not established** | Freeze hard negatives and verify unimodal controls on train only |
| Full media acquisition / model fitting | **Not started** | Disk capacity and current task/data gates do not justify it |

The [CPU waveform audit](R2_FAIRPLAY_AUDIO_PAIR_ALIGNMENT_20261022.md) compares
only eight already acquired train WAV/MP4 audio streams. It does not establish
physical AV synchronization, original-scene grouping, task quality, or that a
Decision task requires both modalities. Keep FAIR-Play blocked from quality
fitting until grouping and both-input necessity are evidenced. Do not read test
IDs/media or select on exposed validation.

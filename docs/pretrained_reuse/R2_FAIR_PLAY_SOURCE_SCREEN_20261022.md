# R2 FAIR-Play metadata-only source screen — 2026-10-22

## Decision

**REVIEW; not admitted for model fitting yet.** FAIR-Play is a plausible, narrowly scoped real audio-video correspondence source: a fixed query audio plus four candidate videos, with the paired video as the positive. It can test whether a frozen representation supports a learned audio/video matching Decision. It does not provide a general multimodal QA benchmark, and any result would be limited to music performances recorded in one room. The current screen does not establish scene-disjointness from raw media, usable hard negatives, or a feasible media subset, so no model training or media download is authorized by this screen.

## Pinned source and rights

- FAIR-Play repository revision: `2666d6caf2d126ae43a84841e94e0c9ae24b3e2d`.
- PseudoBinaural split metadata revision: `87f7be08d41748995c561ad1dc7bda453d4ee553`.
- FAIR-Play's own README describes 1,871 paired video/audio clips, roughly aligned by matching index, an approximately 100 GB full release, and states CC BY 4.0 for FAIR-Play. Treat the asset license as FAIR-Play's grant, with attribution and applicable conditions.
- PseudoBinaural describes five harder, scene-separated splits. Its CC BY 4.0 statement is for the PseudoBinaural software; it is not treated as a separate license grant for FAIR-Play media.
- Primary references: [FAIR-Play repository and dataset license](https://github.com/facebookresearch/FAIR-Play), [2.5D Visual Sound paper](https://arxiv.org/abs/1812.04204), [PseudoBinaural repository and split description](https://github.com/SheldonTsui/PseudoBinaural_CVPR2021), [PseudoBinaural CVPR paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Xu_Visually_Informed_Binaural_Audio_Generation_without_Binaural_Audios_CVPR_2021_paper.pdf).

## Metadata inspected

Only training and validation split metadata was downloaded and parsed. After that metadata audit, exactly one train pair (`000383`) was downloaded into a separate external attempt directory and fully decoded as an input-path preflight. No validation media, official test split, or sealed audit was opened. The original metadata-only receipt remains external to Git:

`C:\CodexArtifacts\pretrained-reuse-r2\fair-play-split-audit-attempt-01\metadata-only-screen.json`

SHA-256: `c6c5c04eb26b56e7e73f0555f9bb537765fe2d48c4290b6039e050f0a8eedb2d`.

At the pinned FAIR-Play revision, each of its ten original random partitions lists 1,497 train and 187 validation IDs with zero within-partition ID overlap. These are re-partitions of the same 1,871 assets, not independent folds (union across folds: 1,871 train IDs and 1,208 validation IDs). Do not combine their metrics.

At the pinned PseudoBinaural revision, each of five `new_splits` partitions lists 1,497 train and 187 validation IDs, with zero within-partition ID overlap. These are also re-partitions of the same asset pool (union across folds: 1,866 train IDs and 744 validation IDs). Select one split only; other folds are not independent replications. The authors state the new splits reduce scene overlap, and their paper describes scene-grouped construction. This screen did not independently establish scene grouping from media or source scene IDs.

The receipt contains per-list byte counts, row counts, ordered ID lists, and SHA-256 digests. Split-list validity is not media integrity: file existence, decoded duration, alignment, scene identity, and normalized media hash overlap remain unknown.

### One-pair train media preflight

- Source: `facebookresearch/FAIR-Play@2666d6caf2d126ae43a84841e94e0c9ae24b3e2d`, ID `000383`, included in the inspected PseudoBinaural split1 train list and original FAIR-Play split1 train list.
- Downloaded only `videos/000383.mp4` (56,478,825 bytes) and `binaural_audios/000383.wav` (1,921,102 bytes). Both SHA-256 values matched their Git LFS object IDs exactly.
- PyAV 18.1.0 fully decoded all streams: video H.264, 1280×720, 60000/1001 fps, 9.5429 s, 572 frames; embedded AAC stereo 48 kHz, 9.984 s, 468 audio frames; paired WAV PCM s16le stereo 48 kHz, 10.0053 s, 118 audio frames. The paired-audio and embedded-audio duration delta was 21.3 ms for this one clip. This validates access and decode for one train pair, not global synchronization or scene grouping.
- External output: `C:\CodexArtifacts\pretrained-reuse-r2\fair-play-train-media-preflight-attempt-01\`; machine-readable receipt SHA-256 `4f43989c2f83684f2b37faf7c3f3bc3f44a6d54d90b35719812973236528a82c`.
- C: free space was 3,579,658,240 bytes before and 3,512,111,104 bytes after. One sampled validation pair's LFS pointers imply about 55.7 MB for its video+paired audio, but its media was not downloaded; this small sample is not a full-split size estimate. No model was loaded and no feature extraction, GPU work, fitting, or quality evaluation was performed.

## Candidate Decision task (not yet frozen or run)

A possible controlled task is: given one audio clip and four candidate videos, return the probability distribution over which video is the corresponding performance. The positive is the same-index FAIR-Play pair. Distractors must come from the same preselected split and, if a verifiable scene mapping can be built, from different scenes. Candidate ordering must be deterministic from a recorded seed and shuffled independently of labels. Freeze the train/validation item manifests before extraction.

Required controls before fitting:

1. Confirm media rights and that individually fetching a small declared subset complies with the FAIR-Play grant and source terms.
2. Verify exact audio/video index correspondence, decodability, duration, and media hashes for all selected train/validation examples.
3. Establish scene grouping independently; reject any group crossing train and validation. If scene identities cannot be verified, do not call the result scene-disjoint.
4. Predeclare candidate construction and ensure each validation audio has one positive and three hard negatives. Random-only distractors are insufficient for a meaningful correspondence test; use only negatives supported by verified metadata/media labels, with no hand-tuned validation selection.
5. Compare audio-only, video-only, frozen-feature mean/concatenation, and a small learned joint compatibility head on identical examples. Include paired stream permutation/replacement controls. A fusion claim requires joint improvement over the strongest unimodal/control baseline and sensitivity to valid pair correspondence.
6. Keep exactly one PseudoBinaural partition (provisionally split1) for this experiment; leave all official test entries untouched.

## Practical gate and next action

At the check on 2026-10-22, local C: free space was approximately 3.27 GiB after retaining the one-pair train preflight, while the full dataset is described as approximately 100 GB. No model features were extracted, and the local RTX 3080 was occupied by another process. Do not launch bulk media acquisition, extraction, or GPU training until the selected subset's actual media sizes and the available encoder path are verified. No existing model, run, checkpoint, or teacher was changed.

If the metadata-to-media preflight clears these gates, proceed only with a small CPU/metadata dry-run and a new external output directory, then use the cooperative local GPU reservation path for a bounded frozen-feature experiment. Report it strictly as narrow music-performance A/V matching; it cannot satisfy the broad four-modality or 95–98% product target by itself.

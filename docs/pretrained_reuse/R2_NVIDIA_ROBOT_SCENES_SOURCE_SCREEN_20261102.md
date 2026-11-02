# NVIDIA PhysicalAI Synthetic Embodied Robot Scenes — R2 source screen

Date: 2026-11-02. Status: `REVIEW_NOT_ADMITTED`. This is a pinned public-metadata screen only. No tar shard, video, sidecar, task label, or model checkpoint was downloaded or inspected; no model inference, fitting, or GPU work occurred. No license acceptance was submitted.

## Pinned source and inventory

- Dataset: [NVIDIA PhysicalAI WorldModel Synthetic Embodied Robot Scenes](https://huggingface.co/datasets/nvidia/PhysicalAI-WorldModel-Synthetic-Embodied-Robot-Scenes/tree/b86a9fe1076e49a2847d95fee4531f541c5b126b).
- Hub revision: `b86a9fe1076e49a2847d95fee4531f541c5b126b` (API `gated=false`; Hub metadata license field `other`). The pinned card says OpenMDW 1.1 and describes the release as commercially and non-commercially usable.
- Public `manifest_public.jsonl`: 227,587 bytes, SHA-256 `507652f9b9f4267c0f2d6d2caac048528059942c0c811ec71b77d38092a7589d`.
- Manifest inventory: 377 WebDataset shards, 373,703 sample records, 2,077,661,327,360 advertised tar bytes (about 2.08 TB). The manifest was read into memory for counts; it was not retained as a local dataset artifact.

| Source group | Shards | Samples | Advertised tar bytes |
| --- | ---: | ---: | ---: |
| collision/isaaclab | 93 | 92,434 | 88,565,565,440 |
| collision/mobilitygen | 3 | 2,080 | 3,556,321,280 |
| manipulation/dreamzero | 23 | 22,858 | 15,170,908,160 |
| manipulation/mimicgen | 70 | 69,212 | 420,346,808,320 |
| manipulation/simulario | 17 | 16,384 | 20,978,677,760 |
| motion/soma | 171 | 170,735 | 1,529,043,046,400 |

The smallest complete source group is `collision/mobilitygen` at approximately 3.31 GiB, larger than the 2.14 GiB C: free-space snapshot during this screen. Individual shards are also large; the first IsaacLab shard advertises 1,162,178,560 bytes. No payload retrieval was attempted.

## Task and split fit

The official card describes RGB MP4 clips and JSON metadata, with task family, generator family, embodiment, scene identifier, task text or motion name, camera setup, frame rate, clip length, and simulator state where available. It names possible state fields such as robot/joint/end-effector/object pose, contact tags, and task-success flags. This could support a video-plus-state task if a reliable target and matched controls can be frozen from a source group.

The release is **video-only** at the declared modality level; it does not provide synchronized audio. It therefore cannot satisfy the current ROADMAP R2 gate for a rights-cleared, source-compatible synchronized audio/video decision task. The card describes organization by scene/layout/task/episode and metadata scene identifiers, and manifest keys visibly contain layout/scene-like tokens. However, no sidecar payload was inspected, the completeness/semantics of scene IDs are unverified, and the release manifest contains shard inventory rather than official train/validation splits. A leakage-resistant scene split has not been built or validated. The card describes generated behavior and simulator metadata, not a frozen typed multiple-choice Decision benchmark; task labels, option construction, class balance, and both-input necessity remain unverified.

## Rights and disposition

The [dataset card](https://huggingface.co/datasets/nvidia/PhysicalAI-WorldModel-Synthetic-Embodied-Robot-Scenes/tree/b86a9fe1076e49a2847d95fee4531f541c5b126b) names [OpenMDW-1.1](https://github.com/OpenMDW/OpenMDW/blob/main/1.1/LICENSE.OpenMDW-1.1). The license states that exercising granted rights accepts its terms and assigns the user responsibility for third-party rights diligence. The card also identifies third-party source materials, including SAGE, SceneSmith, BONES-SEED, and HDRI assets. Although the card claims commercial readiness and names some CC0 HDRI sources, those statements alone do not establish the rights chain for every included asset or for this project's intended use. No license acceptance or third-party-rights investigation was made. Keep this source unadmitted pending rights review.

**Decision:** do not acquire or use payloads, and do not start a model experiment from this screen. The source is a possible future video/state candidate, not an audio/video candidate. Reconsider only after the rights gate, exact scene-key semantics, task/label audit, an independently grouped split, and storage preflight all pass. This does not change or replace the existing Path A baseline and does not justify R3/shared-backbone work.

## Reproduction and limits

The screen used the public Hugging Face dataset metadata API at the pinned revision and fetched only `manifest_public.jsonl` into process memory. The inventory was computed by summing `sample_count` and `tar_bytes` by `shard_group`; no tar or media bytes were requested. Source/model runtime, performance, and parameter counts are not applicable because no checkpoint was loaded. Cloud cost and GPU time were zero.

Official sources: [dataset card and format/organization](https://huggingface.co/datasets/nvidia/PhysicalAI-WorldModel-Synthetic-Embodied-Robot-Scenes/tree/b86a9fe1076e49a2847d95fee4531f541c5b126b), [OpenMDW-1.1 license text](https://github.com/OpenMDW/OpenMDW/blob/main/1.1/LICENSE.OpenMDW-1.1).

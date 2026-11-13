# R2 AVID-Bench source screen (2026-10-21)

## Decision

**REVIEW / not admitted to Path A training or evaluation.** AVID-Bench is a strongly relevant task concept: detecting audio-visual inconsistencies has examples where audio and video must be compared. The currently verifiable release path is not ready for this local project because its media rights and training access are unresolved, and the video volume is far beyond the current local storage budget. This is a source screen only; no AVID data or test labels were downloaded or inspected, and no model was run.

## Evidence inspected

- The official AVID-Bench repository is pinned at `czx1220/AVID-bench@043de166d637a5eae782d01675742640538e253b`. Its README describes 11.2K long-form videos, 39.4K annotated inconsistency events and 78.7K clips across temporal shift, lip sync, voice identity, semantic divergence and related categories. Those labels make this a better *candidate task definition* for joint audio-video reasoning than event visibility alone.
- The repository's Apache-2.0 license covers its code. Its README directs data acquisition to ModelScope but does not state a separate license for the underlying video/audio media or establish rights to the source recordings. The Hugging Face dataset card also labels the dataset Apache-2.0, but that label alone does not resolve rights in the embedded third-party media.
- The official README estimates about 61 MB for annotation files, about 84 GB for the test videos, and about 415 GB for the full dataset. The live C: volume had 19.91 GiB free at screen time, so neither the 84 GB test media nor full training release fits. Do not download the test material.
- The pinned Hugging Face dataset metadata at `Chen-15/AVID@6f36554fa40ea8dc379de54040e165a13fe89e89` exposes test annotation files only in its current file listing. Test annotations are not opened or used; this listing is recorded solely as an access/provenance observation. The GitHub README describes train annotations in the ModelScope release, but this screen did not verify their availability, immutable revision, or media terms there.
- The official baseline fine-tunes Qwen3-Omni with six A100 80 GB GPUs. That baseline is not a compute estimate for our small readout, but it confirms the released evaluation path is not directly comparable to this lightweight feature-reuse setup.

## Gate to reopen

Reopen only after an authoritative source establishes permissible local model training use for the exact source media, a hash-pinned TRAIN-only annotation release is accessible, and a small rights-cleared TRAIN subset can be obtained without touching test media or exceeding local storage. Then inspect scene/video grouping and pair construction before deciding whether to admit a bounded task pilot. Keep the official test sealed. Do not use AVID's reported Qwen scores as Tiny Omni Decision performance.

## Primary sources

- [Official AVID-Bench repository](https://github.com/czx1220/AVID-bench/tree/043de166d637a5eae782d01675742640538e253b)
- [Official repository license](https://github.com/czx1220/AVID-bench/blob/043de166d637a5eae782d01675742640538e253b/LICENSE)
- [Official repository data/download and task description](https://github.com/czx1220/AVID-bench/blob/043de166d637a5eae782d01675742640538e253b/README.md)
- [Hugging Face dataset card](https://huggingface.co/datasets/Chen-15/AVID)
- [Pinned Hugging Face dataset metadata](https://huggingface.co/api/datasets/Chen-15/AVID/revision/6f36554fa40ea8dc379de54040e165a13fe89e89)
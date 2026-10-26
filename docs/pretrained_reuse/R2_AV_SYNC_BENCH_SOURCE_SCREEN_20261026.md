# AV-SyncBench source and task screen — 2026-10-26

## Decision

**REVIEW / not admitted for fitting, Path A quality selection, or the locked
release audit.** AV-SyncBench is directly relevant to temporal and semantic
audio/video correspondence, but it is an evaluation benchmark for pairwise
original-versus-perturbed scoring, not a supplied-option task or training
corpus. Its released media derive from third-party videos; the source materials
do not establish that the benchmark license clears those underlying rights.
The full download is 68.19 GB, while this machine has about 10.9 GB free. The
benchmark and its public paper results are already exposed. No data, media,
model weights, or benchmark labels were downloaded or evaluated.

## Pinned official source inspection

- Official repository: [`Duck-Boss/AV-SyncBench@82e13536074377b71a51d81b2f5559a43cf4a122`](https://github.com/Duck-Boss/AV-SyncBench/tree/82e13536074377b71a51d81b2f5559a43cf4a122).
- The pinned README reports 3,269 in-the-wild videos and 38,390 evaluation
  samples across ten voice, music, and environmental-sound scenarios. It tests
  temporal offset/jitter/speed and semantic timbre replacements. It specifies
  no model training or fine-tuning.
- The official reproduction guide pins ModelScope dataset revision
  `86c06579529a6e7b2cafb0dc386a50152a37fb98`, seven archive parts totaling
  68,191,839,983 bytes, and a 15,000-pair global-offset subset sourced from
  1,500 videos. These are public benchmark materials, not a blind audit.
- The pinned third-party notice says the ModelScope card's MIT declaration
  applies to repository metadata and does not replace rights or platform terms
  for source video. It also records that Synchformer's checkpoint lacks a
  separate published weight license and ImageBind is CC BY-NC-SA 4.0.
- The evaluation rule compares a model score for the original audio/video pair
  against a score for the same video with perturbed audio. This is useful as a
  synchronization diagnostic, but does not by itself test conditional choices,
  typed action probabilities, modality-native Decision behavior, or the
  product's multi-input task battery.

## Local access and receipt hashes

Only small text files were fetched in memory from the pinned official GitHub
revision to verify the source description and hash the inspected files. No
repository clone, dataset archive, source media, or checkpoint was fetched.

| Pinned file | SHA-256 |
|---|---|
| `README.md` | `f6071cbace81ea8ee18cddace710f0251efb296c3007e655a91b303b2f2c580d` |
| `THIRD_PARTY_NOTICES.md` | `d55db4ddee7b025385c8091de24deb92bd54b1f4efe3bf5442aec41f7b003ef2` |
| `docs/DATA_FORMAT.md` | `a353c16c4275fcf374d0413c98bb3ae1339a6d8b59dc316f74db5e6282af9bad` |
| `docs/REPRODUCE_GLOBAL_OFFSET.md` | `5ce1f0d0bb6fe553faca5696587dca9aa24455c129f1d1dbcd79e6f8fc9f68cb` |
| `docs/MODEL_ZOO.md` | `5b1ccbda2a9222ebb8b0a13ba8d7a2d74fbed4470e94c7a89fd4bfe3a7e28cb9` |

## Re-entry conditions

Reconsider this benchmark only if the original-media rights and applicable
source-platform terms are affirmatively cleared for the intended model use, a
separate storage plan exists, and a compact, rights-cleared scorer is selected.
Any later use must remain a labeled synchronization diagnostic, not a
substitute for independently grouped Decision tasks or the locked final audit.

## Sources

- [Official AV-SyncBench repository at pinned revision](https://github.com/Duck-Boss/AV-SyncBench/tree/82e13536074377b71a51d81b2f5559a43cf4a122)
- [Official reproduction guide at pinned revision](https://github.com/Duck-Boss/AV-SyncBench/blob/82e13536074377b71a51d81b2f5559a43cf4a122/docs/REPRODUCE_GLOBAL_OFFSET.md)
- [Official data format at pinned revision](https://github.com/Duck-Boss/AV-SyncBench/blob/82e13536074377b71a51d81b2f5559a43cf4a122/docs/DATA_FORMAT.md)
- [Official third-party notices at pinned revision](https://github.com/Duck-Boss/AV-SyncBench/blob/82e13536074377b71a51d81b2f5559a43cf4a122/THIRD_PARTY_NOTICES.md)

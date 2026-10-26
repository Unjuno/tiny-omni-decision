# M³AV audio-visual lecture source screen — 2026-10-26

## Decision

**REVIEW / not admitted for Path A fitting or product-quality evaluation.**
M³AV is a technically relevant audio/slide candidate: the publisher reports
1,113 lecture videos (about 367 hours), word-level speech annotations, slide
OCR, and a speaker-disjoint train/development/test split. However, its
CC BY-NC-SA 4.0 terms are research-only for noncommercial purposes, while the
publisher explicitly says video copyright remains with original owners. The
45.5 GB hosted dataset also exceeds current local free storage. Do not acquire
or use the full media under the current product-model objective without a
component-level rights path and a storage plan.

## Pinned source inspection

- Official project repository: [`Jack-ZC8/M3AV-dataset@1aa6fa8992fe0bdd28583d3ad997a28f1cc18c03`](https://github.com/Jack-ZC8/M3AV-dataset/tree/1aa6fa8992fe0bdd28583d3ad997a28f1cc18c03).
- Publisher download instructions say the dataset is licensed CC BY-NC-SA 4.0
  for noncommercial purposes and that the video copyrights remain with the
  original owners. The repository's `LICENSE` and dataset-specific instructions
  both identify CC BY-NC-SA 4.0.
- Hugging Face dataset card identifies `cc-by-nc-sa-4.0` and lists 45.5 GB total
  file size: [CHHHH/M3AV_v1.0](https://huggingface.co/datasets/CHHHH/M3AV_v1.0).
- The paper reports five video sources (CHI, UbiComp, NIH, IPP, Oxford
  Mathematics) and contextual-ASR, TTS and slide/script-generation tasks. The
  source license does not itself establish reuse permission for every original
  lecture recording.

The official `download/set_split.json` contains 1,113 video IDs: 901 train, 99
development and 113 test. By source prefix, the counts are:

| Source | Train | Development | Test |
|---|---:|---:|---:|
| CHI | 547 | 55 | 58 |
| IPP | 49 | 5 | 13 |
| MLS | 41 | 6 | 4 |
| NIH | 183 | 23 | 22 |
| UbiComp | 81 | 10 | 16 |

The publisher download README says this split follows speech-speaker
assignments; the paper describes no speaker overlap across splits. This is
useful split design, but no local media-to-ID or source-rights audit has been
performed. Publicly reported dev/test benchmark results mean those sets should
not be treated as sealed audits.

## Local source receipt and access boundary

The source commit and metadata hashes were recorded during a shallow Git source
inspection:

| File | SHA-256 |
|---|---|
| `README.md` | `eabd9dbe8cdc4cc531cc3c0c3a4ce95c587e205b7cca5e78bb54a0579db279d6` |
| `LICENSE` | `047d2259741a3ebb30d8c8a43d4ba79b5b229a069acd1d2bea49f22b297d8e98` |
| `download/README.md` | `4221af718874794f56018fb800183ee7ce8b1f95e30a11ed398e9d0b464f2e3a` |
| `download/set_split.json` | `5fd6a4460b4db6e5e487814ec2252596d5476e2bd7a3e974715a8fd62945c0f0` |
| `download/M3AV_v1.0_sha256.txt` | `ca7d9c68e83cbb89fa41f38f26824bb7e179927b988ea9c2993d3c5bd0270e22` |

The shallow source clone included publisher-supplied demo images and sample
metadata, including annotation examples, as ordinary Git-tracked files. Those
demo assets were fetched but not opened or used; cleanup removed them and the
checkout's working files. The local shell policy blocked deletion of residual
`.git` pack files, leaving about 22.1 MB under
`C:/CodexArtifacts/pretrained-reuse-r2/m3av-source-audit-20261026/.git/`.
No full dataset archives, raw lecture recordings or model features were
downloaded. The `set_split.json` metadata was read to count IDs by split and
source prefix. No model was run or trained.

## Possible bounded task if rights and resources change

A research-only Decision adaptation might compare audio-only against
audio-plus-slide evidence for choosing a supplied technical term from candidate
words around a spoken rare-word timestamp. It would require a leakage-safe
target/distractor constructor, inspection of training-only annotations,
pretraining-overlap checks for the pinned encoders, and source-specific rights
clearance. This would be a narrow contextual speech Decision, not general video
understanding or the full Omni target. Do not use the public test split as a
blind audit.

## Sources

- [Official M³AV project page and benchmark description](https://jack-zc8.github.io/M3AV-dataset-page/)
- [Official source repository](https://github.com/Jack-ZC8/M3AV-dataset)
- [Official dataset card and file-size/license metadata](https://huggingface.co/datasets/CHHHH/M3AV_v1.0)
- [ACL 2024 paper](https://aclanthology.org/2024.acl-long.489/)

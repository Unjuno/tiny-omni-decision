# MAVERIX audio-visual reasoning source screen — 2026-10-26

## Scope

Metadata-only source admission screen. No question/answer JSONL, subtitles,
video/audio, weights, or predictions were downloaded or opened. No benchmark
inference, training, or test inspection occurred.

The [official MAVERIX site](https://maverix-benchmark.github.io/) describes
700 videos, 2,556 QA pairs, four reasoning task groups, and heterogeneous
video/audio/subtitle inputs. It reports public model and human results and a
2025-09-10 opened test submission server; therefore published results and
benchmark test material are exposed and cannot be treated as a new blind audit.

The official site links [code](https://github.com/CUBE-CMU/maverix) and
[Hugging Face data](https://huggingface.co/datasets/CUBE-CMU/maverix). A
metadata-only query pinned HF revision
`94e19b1c2575d2b96ad52b3e9c9ff05753e75415`, with four file entries (2,217,630
bytes total): `.gitattributes`, `maverix.py`, `maverix/test.jsonl`, and
`subtitle.tar.gz`. The tree has no video or audio media files and no license
field was declared. Because no dataset payload/card was opened, the exact
question schema, clip links, underlying media rights, and any non-test partition
are **UNKNOWN**. The listed `test.jsonl` itself was not read.

The source screen queried metadata only. A receipt is stored outside Git at
`C:\CodexArtifacts\pretrained-reuse-r2\maverix-source-screen-20261026\receipt.json`
(SHA-256 `807c81cf010a28f0712d1f79c0ccc8d41051a9972ef2907f8a8babaa18776df5`).
It records the pinned revision, four file entries, no declared license, and that
no payload/annotations were downloaded/opened.

## Disposition

**REVIEW / not admitted** for fitting, candidate selection, product evaluation,
or packaging. MAVERIX is a useful AV task-design lead, but its metadata-only
release lacks the media needed for a real audio/video model test; its media
rights and permissible access are unresolved, and the public benchmark has
already been scored. Do not open its QA file or retrieve linked media until an
authorized source, license basis, independent grouped development split, and
sufficient local storage are established. Continue with the rights-cleared
VisTouch lead if the official native archive becomes available in the workspace.

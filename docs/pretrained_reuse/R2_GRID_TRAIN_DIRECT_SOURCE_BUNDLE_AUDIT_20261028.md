# GRID train direct-source bundle audit (2026-10-28)

Status: **source inventory verified for all 24 frozen train speakers; the
frozen train speaker/code mapping remains blocked for training.** Official
per-speaker audio, word-alignment and video filename inventories agree for
all 24 speakers. The frozen train manifest matches the official same-label
source inventory for only 9 speakers; 15 speakers have zero code overlap.
No existing split, media, model, or checkpoint was modified, and no speaker
identity was inferred or remapped.

## Scope and method

This completes the direct-source check required by the earlier
[`GRID train partition source-mapping audit`](R2_GRID_TRAIN_PARTITION_SOURCE_MAPPING_AUDIT_20261028.md).
The authoritative frozen train input is
`grid-development-split-20261028-attempt-02/train.json`, SHA-256
`f6f045931aee53e576fca22efda0bc888a983aed685a30edc38b2918ff156817`.
The same 24-speaker train partition was retained. Each speaker's official
Sheffield audio TAR and alignment TAR were downloaded and SHA-256 hashed; the
final 262,144 bytes of the corresponding official Zenodo video ZIP were
fetched with a validated HTTP 206 `Content-Range` to parse its central
directory. The existing
[`audit_grid_direct_speaker_codes.py`](../../scripts/audit_grid_direct_speaker_codes.py)
compared the three six-character code inventories against the frozen train
manifest. Its focused tests passed (3 passed); Ruff passed for the audit code
and tests.

No audio or video media payload was decoded or semantically inspected. The
complete audio/alignment TAR files were downloaded and hashed; video bodies
were not downloaded. The audit is a file/code correspondence check only. It
does not establish semantic content, cross-corpus speaker independence,
product distribution rights, task quality, or model performance.

## Results

All 24 direct-source bundles had exactly 1,000 audio codes, 1,000 alignment
codes, and 1,000 video codes, with pairwise intersection of 1,000. The frozen
train manifest contains 23,992 rows; the official same-label overlap was:

| Finding | Speakers | Rows / codes |
| --- | --- | ---: |
| Direct bundle matches frozen split | s16, s17, s2, s3, s31, s34, s7, s8, s9 | 8,995 matching rows |
| Direct bundle matches each other, frozen split has no same-label code overlap | s10, s11, s12, s14, s15, s19, s20, s22, s26, s28, s29, s32, s33, s5, s6 | 14,997 frozen rows mismatched |

The 15 mismatched speaker groups are distributed across the original train
partition. They cannot be treated as valid train examples for the intended
audio/video correspondence task. Do not repair the frozen manifest by
permuting labels from the overlap matrix. A future corpus must be generated
from the authoritative per-speaker source bundles, with a fresh manifest,
new split hashes, and the existing speaker-disjointness and media-overlap
gates rerun before any training.

## Receipts

External data and per-speaker receipts remain outside Git:

- Attempt 01: `C:/CodexArtifacts/pretrained-reuse-r2/grid-train-direct-speaker-audit-20261028-attempt-01/`; `acquired.jsonl` SHA-256 `f03a166157c728c8d58754e1b0b1b2d01bfb2d0a02fd51dc7701045379dd9f34`.
- Attempt 02: `C:/CodexArtifacts/pretrained-reuse-r2/grid-train-direct-speaker-audit-20261028-attempt-02/`; `acquired.jsonl` SHA-256 `5ec81b529f4a504d3f03066cdebe466edc7b35db067155e4406a7846c038f4c4`.
- Attempt 03: `C:/CodexArtifacts/pretrained-reuse-r2/grid-train-direct-speaker-audit-20261028-attempt-03/`; `acquired.jsonl` SHA-256 `a7f5812e7182882e10ddcbf13a5e45f90a1fd61b2d2011de48e1b8a853d4dc2f`.
- Combined 24-speaker result: `train-direct-audit-summary.json` SHA-256 `34450c209ec5e45d2025ff09eb4227d366cce7eee6459eacffd383e022f42eb6`.
- Combined totals: 2,275,911,680 bytes of audio TARs, 24,821,760 bytes of alignment TARs, and 6,291,456 bytes of video ZIP metadata tails.

The first acquisition attempt included three already-matching speakers and
omitted three mismatching speakers due to an incorrect hand-copied list. Its
files were preserved. The frozen mapping JSON was then used to derive the
exact remaining set; attempts 02 and 03 acquired and audited those missing
speakers. The final combined summary verifies exact 24/24 coverage against
the frozen split and separates all three attempts without overwriting any
receipt.

## Consequence and next gate

The source files themselves are coherent, but the existing train manifest is
not. Do not train on the frozen GRID split or the failed source-mapping
corpus. Rebuild a new train/validation manifest directly from the official
per-speaker bundles and annotations, retaining the fixed speaker partition
unless a newly justified split protocol is documented. Verify the rebuilt
manifest and all source/media overlap gates before generating features or
starting a model run. GRID remains internal research-only pending product
rights clarification; this audit does not change that boundary.

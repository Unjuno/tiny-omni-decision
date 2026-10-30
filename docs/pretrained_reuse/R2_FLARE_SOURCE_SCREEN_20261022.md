# R2 FLARE source and task screen — 2026-10-22

## Disposition

**REVIEW / NOT ADMITTED.** FLARE is unusually well aligned with a joint audio-video
retrieval diagnostic, but its released video clips originate from Video-MME,
whose terms prohibit commercial use and restrict copying and distribution. The
FLARE dataset-level CC BY 4.0 declaration does not establish rights to those
underlying videos. No dataset archives or media were downloaded or locally read,
and no model was run.

## Task fit

FLARE describes 399 long-form videos (225.4 hours), segmented into 87,697
audio-bearing clips, with 274,933 text queries. It reports 53,580 cross-modal
queries filtered so vision-only retrieval and audio-only retrieval fail while
the joint audiovisual retrieval succeeds. That makes FLARE a strong *task
design lead* for testing complementary AV evidence.

It is a text-to-clip / text-to-video retrieval benchmark, however, rather than a
ready-made supplied-option Decision task. The reported hard-bimodal condition is
defined by retrieval-system ranks; it does not itself provide a calibrated
multiclass target distribution or establish a production decision policy.
Converting it into fixed candidate options would require a preregistered gallery,
negative construction, scoring protocol, and video-grouped split design.

## Rights and split evidence

- FLARE's official project page says the benchmark was built from 399 screened
  Video-MME videos and declares the dataset CC BY 4.0; its official repository
  separately declares code MIT.
- The FLARE Hugging Face dataset card lists video media as 13 ZIP volumes in
  addition to text annotations. The viewer showed a single `test` split for the
  selected 87,697-row clip-caption subset. No separate train/validation/audit
  split or video-disjoint partition was established by this metadata screen.
- At this review, the Hugging Face API reported FLARE revision
  `ed89d868647a3658fe6989ba06b7a7117c2624d9`, last modified 2026-05-07, with
  `cc-by-4.0` metadata. The Video-MME repository revision reviewed was
  `06c2315b892f88578f81d73205d07cf576f292b9`.
- Video-MME's official README says academic research only, prohibits commercial
  use, identifies video copyright as belonging to the source owners, and says
  the benchmark may not be distributed, copied, disseminated, or modified
  without prior approval. The FLARE release includes the segmented MP4 clips,
  so the upstream rights question is material even when considering local
  research evaluation.

No assumption is made that the FLARE CC BY declaration overrides Video-MME's
terms or any original video owner's rights. During this public-source screen,
the Hugging Face viewer displayed a small number of example caption rows; these
were not saved or used. Do not download, train on, evaluate against, or package
FLARE media unless the applicable rights are documented. A separately
authorized subset would still need source-owner and video-ID provenance, media
hashes, leakage checks against all prior corpora, and independently grouped
train/validation/audit splits.

## Decision for the roadmap

Keep FLARE as a high-priority task-design reference, not an admitted quality
source. Do not use its `test` viewer subset for repeated candidate selection.
Continue searching for a synchronized AV source with explicit training and
development authorization, video-grouped partitions, and labels that support a
fixed supplied-option task. If FLARE authors provide written rights covering
the underlying media and a defensible split, re-screen those materials before
access.

## Sources

- FLARE project page: <https://flarebench.github.io/>
- FLARE official repository: <https://github.com/YqjMartin/FLARE>
- FLARE dataset card: <https://huggingface.co/datasets/AnonymousFLARE/FLARE>
- FLARE metadata API (revision above): <https://huggingface.co/api/datasets/AnonymousFLARE/FLARE>
- Video-MME official README/terms: <https://github.com/MME-Benchmarks/Video-MME/blob/main/README.md>
- Video-MME repository revision: <https://github.com/MME-Benchmarks/Video-MME/commit/06c2315b892f88578f81d73205d07cf576f292b9>

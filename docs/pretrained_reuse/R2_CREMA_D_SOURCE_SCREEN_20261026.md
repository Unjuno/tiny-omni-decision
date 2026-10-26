# CREMA-D audio-visual source/task screen — 2026-10-26

## Disposition

**REVIEW; not admitted for training or quality claims yet.** CREMA-D is a
stronger source lead than generic audiovisual event co-occurrence for testing
whether frozen audio/video representations help predict human-perceived
emotion: it contains separate audio-only, video-only, and joint audiovisual
ratings for the same acted clips. However, its task is narrow acted-emotion
recognition, not general Omni Decision, and the available rights declaration
does not resolve performer consent/personality-rights scope for model training.
Do not download the multi-gigabyte media or fit a model until that remaining
rights question and a leakage-safe split are resolved.

## Pinned source and observed facts

- Official source repository: `CheyneyComputerScience/CREMA-D`, `master` at
  `1658cd342dff90010aa843eaeebd53610a08b1dc` (read-only `git ls-remote`,
  2026-10-26).
- Its README describes 7,442 clips from 91 actors, 12 sentence prompts, six
  intended emotions, and 2,443 human raters. Raters assessed voice-only,
  face-only, and audiovisual presentations, with separate modality-specific
  emotion and intensity labels. The repository provides audio, video, rating
  tables, and actor demographics. It asks repository users to fill out a
  usage-tracking form; no form was submitted.
- The official README declares the database under ODbL 1.0 and individual
  contents under DbCL 1.0. DbCL explicitly grants commercial copyright use,
  while ODbL governs database rights and imposes attribution/share-alike
  conditions on publicly used derivative databases. These declarations do
  not themselves resolve rights the licensor may not own, privacy, publicity,
  or performer consent for model training. The paper describes the data as
  acted performances and human perception studies but the checked paper text
  did not specify a model-training consent/release scope.
- The upstream Git repository uses Git LFS; a full clone is documented as
  about 7.55 GB. The repository was not cloned and no media or labels were
  downloaded or opened during this screen.
- TensorFlow Datasets' `crema_d/1.0.0` provides 5,144 train, 738 validation,
  and 1,556 test records grouped so each speaker appears in exactly one split,
  but that TFDS version contains **audio only**. It cannot serve as the joint
  audio/video corpus without separately auditing the original media and
  matching its actor split.
- The paper reports that audio-only emotion recognition is harder than visual
  and audiovisual recognition, and that the relative contribution varies by
  emotion. It also reports substantial ambiguity and intensity dependence;
  accuracy on only high-intensity or agreement-filtered examples would
  overstate broad task quality.

## Potential Path A experiment, only after rights and data gates

If the data owner confirms that the existing license and performer releases
cover model training and the files are accessible through an authorized route:

1. Freeze a manifest of clip IDs, actor IDs, source file hashes, and rating
   table hashes. Derive actor-disjoint train/validation/held-out groups before
   feature extraction; never split clips from the same actor across groups.
2. Keep the already-public labels described as **exposed held-out evaluation**,
   not a blind or sealed audit. Freeze its IDs and evaluate only after model and
   readout selection.
3. Use the release's modality-specific human labels, preserving vote
   distributions/intensity and ambiguous cases where possible. Do not replace
   all targets with filename-encoded intended emotion.
4. Compare frozen audio-only, video-only, and joint features with the same
   actor split, options, and calibration procedure. Predeclare mismatch,
   missing-stream, and modality-shuffle controls. Report per-emotion metrics,
   Accuracy, macro Accuracy, NLL, Brier, ECE, coverage, and actor-grouped
   uncertainty.
5. Treat the outcome solely as a narrow acted-expression perception task; it
   cannot establish natural conversation understanding or general multimodal
   Decision quality.

## Sources

- [Pinned official CREMA-D repository README](https://github.com/CheyneyComputerScience/CREMA-D/blob/1658cd342dff90010aa843eaeebd53610a08b1dc/README.md)
- [CREMA-D paper (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4313618/)
- [TFDS CREMA-D catalog](https://www.tensorflow.org/datasets/catalog/crema_d)
- [Open Database License 1.0](https://opendatacommons.org/licenses/odbl/1-0/)
- [Database Contents License 1.0](https://opendatacommons.org/licenses/dbcl/1-0/)

## Next gate

Before acquisition: obtain an explicit answer from the data owner about model
training/redistribution and performer release scope, or identify a separately
authorized source for the media. Then inspect only metadata to freeze actor
groups and estimate class/modality coverage. If those gates fail, reject the
candidate and preserve this screen; do not substitute its exposed labels or
media into another split.

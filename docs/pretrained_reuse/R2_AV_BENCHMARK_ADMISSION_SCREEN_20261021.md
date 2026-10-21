# R2 audiovisual benchmark admission screen — 2026-10-21

Status: source-screen only. No annotations or media were downloaded or opened,
and no model inference or training was run.

## Purpose

R2 needs a task where synchronized audio and video have complementary, testable
roles. This screen checks two official benchmark candidates before any data
acquisition or model run. A public repository or permissive code license does
not by itself grant rights to underlying media or annotations.

## AVI-Bench

- Official source: <https://github.com/FudanCVL/AVI-Bench>
- Revision checked: `fec7b14bae3eacd5a606847313aa5faf6e428207` (`main`, 2026-10-21).
- The repository describes 14 tasks and 5,864 samples spanning perception,
  understanding, reasoning, and primitive sensation, including AVQA and
  audiovisual grounding tasks.
- `DATA_USE_POLICY.md` v1.0 (last updated 2026-05-28) says the dataset is
  CC BY-ND 4.0 with a binding Anti-Training Addendum. It permits evaluation of
  pre-existing models, but prohibits training, fine-tuning, distillation,
  parameter updates, training-data construction, signal mining, and bulk
  crawling/downloading. Bundled media retains upstream terms.
- **Disposition:** not admissible for any fitting or training-data generation.
  It may be considered for one-time evaluation of a pre-existing frozen model
  only after checking upstream media rights and the exact permitted access path.
  The policy prohibits batch downloading, so do not mirror or locally ingest
  the full benchmark. No model currently qualifies for this use as a product
  evaluation, and this screen does not authorize opening its samples.

## MUSIC-AVQA

- Official code: <https://github.com/GeWu-Lab/MUSIC-AVQA>
- Official project page: <https://gewu-lab.github.io/MUSIC-AVQA/>
- Revisions checked: code `cc8809514a2ee3d7ae81a3021c15e7c0679170d7`
  (`main`, 2026-10-21); project page read on 2026-10-21.
- The project describes roughly 45.9k QA pairs over about 9.3k videos and
  audio-only, visual-only, and AV question types. Its examples include
  identifying the first sounding instrument and associating visible instruments
  with sounds, making it a plausible task-design reference for AV complementarity.
- The project states that real videos were manually collected from YouTube and
  that synthetic videos include random audio/video matching, audio overlays,
  and video stitching. The official code is MIT, but the reviewed project page
  describes the dataset as publicly available for research purposes and does
  not provide a clear dataset/media license grant. The page also gives raw
  archive sizes of 36.67 GB real and 11.59 GB synthetic. The README and project
  page report slightly different total video counts (9,288 vs 9,290).
- **Disposition:** review-only; do not download, manifest, or train on it until
  rights for the annotations and each underlying media component are
  established. Its synthetic mismatch cases could help design controls, but
  must not be treated as valid aligned AV examples without auditing labels.

## Decision and next evidence

Neither candidate is admitted as a training corpus. AVI-Bench offers a clear
evaluation-only pathway subject to access and upstream rights; MUSIC-AVQA is
task-relevant but its reviewed public materials do not establish rights to the
underlying YouTube media. Continue using the already-admitted Perception Test
only as an exploratory source; its existing matched-input results remain
low-quality and do not qualify AV fusion.

Before a new AV fit, require a rights-cleared source, pinned records, explicit
scene/session-disjoint splits, audited alignment and labels, and identical
examples for audio-only, video-only, and joint comparisons. Include missing,
shuffled, and stale-modality controls. Do not infer semantic necessity from
time synchronization or output sensitivity alone.

## References

- AVI-Bench repository and policy: <https://github.com/FudanCVL/AVI-Bench/tree/fec7b14bae3eacd5a606847313aa5faf6e428207>
- MUSIC-AVQA repository: <https://github.com/GeWu-Lab/MUSIC-AVQA/tree/cc8809514a2ee3d7ae81a3021c15e7c0679170d7>
- MUSIC-AVQA official project page: <https://gewu-lab.github.io/MUSIC-AVQA/>

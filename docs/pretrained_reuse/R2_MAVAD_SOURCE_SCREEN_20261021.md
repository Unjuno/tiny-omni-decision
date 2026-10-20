# MAVAD source screen — 2026-10-21

## Disposition

**Promising task shape, but not admitted for training or model selection.** The
official Zenodo API says CC BY 4.0, but the dataset's own metadata PDF says
"Creative commons licence (by attribution, no commercial use)." That is a
material rights conflict for a project with a commercial-use objective. Keep
the data at REVIEW and do not download or train on it unless the rights holder
clarifies the governing terms. Independent source grouping and evidence for
the intended AV task are also unresolved. No video archive or annotation
payload was opened, and no model, feature cache, GPU, or evaluation split was
touched.

## Pinned source and rights evidence

- Dataset record: [Zenodo record 7950008](https://zenodo.org/records/7950008),
  DOI `10.5281/zenodo.7950008`, concept DOI `10.5281/zenodo.7950007`, created
  2023-06-01 and last modified 2024-07-12. The [official Zenodo record API](https://zenodo.org/api/records/7950008)
  returned `metadata.license.id = cc-by-4.0` and `access_right = open` on
  2026-10-21. However, the included metadata PDF's Rights field reads
  "Creative commons licence (by attribution, no commercial use)" and its
  Access Information field separately names CC BY 4.0. Treat the two statements
  as conflicting; the PDF was downloaded, size/MD5/SHA-256 checked, and text
  extracted. This source is **not cleared for commercial use** based on the
  current record alone.
- Paper record: [Zenodo record 8086926](https://zenodo.org/records/8086926),
  DOI `10.5281/zenodo.8086926`; its [API metadata](https://zenodo.org/api/records/8086926)
  also reports CC BY 4.0. The
  paper is separately available as [arXiv:2305.15084](https://arxiv.org/abs/2305.15084).
- The dataset API inventory lists three masked media archives and one metadata
  PDF, totaling **11,978,855,585 bytes** (about 11.16 GiB):

| File | Bytes | Zenodo MD5 |
|---|---:|---|
| `Masked_Mgarr.zip` | 5,498,240,496 | `b47ce8fe6a93d25ae6213410580edb78` |
| `Masked_Zejtun_Field.zip` | 4,659,232,563 | `94fb5d37c362d82ef14785ac6944c8a6` |
| `Masked_Zejtun_Scrapyard.zip` | 1,821,282,219 | `3cca413c94ffc5b6800aa6daa26fafae` |
| `MARVEL_AV_AD_Metadata.pdf` | 100,307 | `e70d6dd2eeeb03936bd9fb9038d807fd` |

The retrieved PDF bytes match the listed size and MD5 and have SHA-256
`83f1d2d671f8f8373e04bae6dfbd55d656b2e8fc9f0215bae371df8ae5a4060e`; the
copy is preserved outside Git at
`C:/CodexArtifacts/pretrained-reuse-r2/mavad-source-screen-20261021/MARVEL_AV_AD_Metadata.pdf`.
Official Zenodo API response snapshots are preserved outside Git in the same
directory. Their SHA-256 values are `B6D9DE88D0E563868E0EACF16707EFDE8806B974E038005DB0B9160677E6B9D4`
(record 7950008) and `6A87865005870786D1632B85C7F53B3E0946E9BE606343F3A5692909B663F3DB`
(record 8086926). These are read-only source metadata artifacts; media files
were not fetched.

The paper describes three camera locations in Malta, 764 video clips, 11
video-level event classes, synchronized stereo audio at 48 kHz, and video at
25 or 30 fps. It says released video is masked to blur faces and license
plates. It says audio was not anonymized because the microphones were distant
and speech/PII was considered unlikely. Treat that statement as source author
context, not as an independent privacy audit. The metadata PDF says annotators
received five-minute snippets and trimmed them into shorter labeled videos; it
does not provide a parent-snippet-to-trimmed-clip ID map. It says each video
filename has the anomaly label plus an instance number. A feature pipeline must
derive IDs from content/provenance, never use label-bearing filenames as model
inputs, and group clips by their original five-minute snippet when those links
can be recovered.

## Task fit and limits

The paper frames the labels as weakly supervised video-level normal/anomaly
classification. Its evaluation repeats per-class randomized 60/20/20 splits
within each scene three times. That is not an independent location split. The
metadata PDF further confirms that multiple shorter clips were cut from
five-minute snippets, so a clip-random split could leak a parent recording.
The included metadata does not map child clips back to those parent snippets.
Before any quality-selected run, an annotation-only inventory must determine
stable clip IDs, parent snippet/session IDs, temporal overlap, class coverage
by group, and whether a practicable train/development/sealed grouping exists.
Keep every child clip from a parent snippet in one partition. With only three
camera locations, leave-camera-out evaluation may be too small or
class-incomplete; do not silently fall back to randomized clip splits.

The paper reports that audio improves its audiovisual method over a zero-audio
variant on each of the three scenes. The reported absolute AUC gains are 1.85,
0.85, and 3.87 percentage points. This is promising redundant-task evidence
from the authors' repeated random splits, but does not establish that a real
decision requires both streams, nor that the gains transfer to independently
grouped data. A project evaluation must include video-only, audio-only,
audio-video, matched and mismatched pairs, zero/replaced-stream controls,
calibration metrics, and grouped uncertainty. It must not reuse the authors'
reported split results as a blind project test.

## Next gate

1. Obtain authoritative clarification of whether the deposit is available for
   commercial model training and derived features; do not infer permission
   from the API license field while the metadata PDF conflicts.
2. Only after the rights conflict is resolved, inspect the annotation files
   and determine if child clips can be joined to parent five-minute snippets.
   Do not download the 11.98 GB media archives until the parent-group structure
   and class-by-group feasibility are known.
3. If a fresh grouped train/development partition is feasible, record immutable
   IDs, source hashes, label validation, and media integrity requirements
   before downloading selected media. Keep an untouched group for a later
   frozen audit, if the source supports one.
4. Define whether the intended task is redundant AV event classification or
   demonstrably both-required reasoning. If only the former is supportable,
   report it as a redundant AV benchmark and do not claim complementary fusion.

No GPU fit is authorized by this screen. Any later local GPU work must use the
repository HotUGPU wrapper, which waits for cooperative yield and starts its
child only after the reservation and free-memory checks pass.

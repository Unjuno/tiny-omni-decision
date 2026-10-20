# CLEVRER scene exposure and synthetic AV feasibility audit — 2026-10-20

## Decision

CLEVRER has enough train-only collision annotation for a bounded synthetic
audio/video timing diagnostic, but it is **not** a natural audio/video source
and this audit does not close R2-M or establish product quality. Do not call
the remaining scene IDs a sealed or proven-untouched split. They are screened
candidates pending content/media identity checks and a predeclared task split.

## Exposure scan

The audit covered the R2 artifact root and all 23 registered Git worktrees.
It scanned 1,023 JSONL files (3,922,086 rows), 4,983 relevant JSON metadata
files (125,637 JSON objects), and video filenames in the same roots. Paths and
filenames marked `test` or `sealed`, and test/sealed/audit directories, were
excluded. Only metadata and media filenames were read; video bytes, validation,
test, and sealed-audit data were not opened.

The local official training question source matched the pinned SHA-256
`11181da673d223f41fb596aacfbbd3ff83d39af7f09a3210549e98cb283714b4` and listed
10,000 train scene IDs. The metadata scan found 9,589 distinct train IDs already
represented. A filename scan found 6,045 CLEVRER scene IDs with local video
files. Removing both sets leaves 405 candidate train scenes. The candidate IDs
are sorted and hashed; the receipt also records hashes for the source metadata
file list, exposed IDs, local media IDs and official train IDs.

Candidate-ID SHA-256: `33bb4408f20ec95a64ed657da5e860588dccfd5378b95c52101ab0768baba4e8`.
The metadata path-list SHA-256 is
`43e4bf91ab38a63f6486854f13b86dfcbec0dfd9efae058038bf36f1dfb2ecb2`; the
exposed-ID SHA-256 is
`5c0d41989f27ec88e8bb46d0223cb69a9a8b6f7b2a4642fc0b365ffd5d4483aa`, and the
cached-media-ID SHA-256 is
`64efadee9f7c8962ddcb5889a6356b7ef71458c0b14a5137805e479f941922f1`.

The prior held-out scene manifest (`clevrer-heldout-video-excluded-scenes-20261011.json`)
contains 307 train and 212 validation scenes; it has zero overlap with the 405
candidates. Exact normalized question text is less clean: of 6,113 source
question rows attached to the candidate scenes, 5,395 rows (2,638 distinct
normalized question strings) also occur in prior project metadata. A
seed-17 80/20 scene-grouped draft partition would share 434 of 917 validation
question strings with train. This is template reuse, not shared video identity,
but it must be disclosed and the AV task must not be answerable from question
text alone.

Normalization for that diagnostic is Unicode NFKC, case-folding, punctuation
replacement with spaces, and whitespace collapse. It compared question text
only. It did not prove zero overlap of programs, choices, answers, or media
content. All CLEVRER questions from one video must remain in one split.

## Train annotation coverage

The official CLEVRER README states that train scenes are 0–9,999, each video is
five seconds at 128 frames, the data is CC0, and train annotations contain
object attributes, trajectories and collision events. It describes four
question families: descriptive, explanatory, predictive and counterfactual.
See the [official dataset README](https://data.csail.mit.edu/clevrer/README.txt)
and [CLEVRER project page](https://clevrer.csail.mit.edu/).

Only the 405 screened train-scene annotation files were fetched from the
official annotation ZIP using HTTP byte ranges. The ZIP advertised size was
156,761,889 bytes with ETag `"957ff21-59db3309a63ea"`; 7,440,931 range bytes
were read. Every ZIP member passed the ZIP CRC check, each `scene_index` was
checked against its requested ID, and each extracted file has a SHA-256 in the
external receipt. The selected annotation payloads total 53,005,401 bytes.
The archive itself was not downloaded, so its full-archive SHA-256 is unknown.
All 405 scenes contain collision events: 1,052 events total, each with two
valid object IDs and a frame index in `[0,127]`. Every event has a pair of
object attribute signatures that uniquely identifies the colliding objects
within its scene. No train video bytes have yet been fetched.

The official train question source associates 6,113 questions with these
scenes: 4,453 descriptive, 688 explanatory, 225 predictive and 747
counterfactual. This inventory was read only from the train split. These
question counts are not an AV evaluation set.

## Reproducibility artifacts

The final metadata-only exposure receipt is:

`C:\CodexArtifacts\pretrained-reuse-r2\clevrer-synthetic-av-exposure-v1-attempt-04\receipt.json`

Receipt SHA-256: `0fc99660e3ea05095856696540ace7bbc5dc9899a4e4b612893a4edb69e65f38`
The audited script SHA-256 is
`860ee9c5c38617baca99c2248d0424e3cf605df82109101393ba325e9adc87f1`.

The annotation extraction receipt and 405 JSON files are under:

`C:\CodexArtifacts\pretrained-reuse-r2\clevrer-synthetic-av-annotations-v1-attempt-01\`

Annotation receipt SHA-256:
`26c178bbfcbc06f38828d47592e7cce7a545c0372f4f9b7f676e7322a6e41485`

The video archive advertised size `12,354,893,389` bytes and ETag
`"2e068b64d-59db378b1c46e"`; a 1 KiB range probe confirmed the server returned
HTTP 206 with the expected range and ETag. No video member was downloaded.

Audit outcomes from exploratory attempts 01–03 are preserved in their separate
artifact directories. Attempt 01 used an overly broad text match and excluded
some scenes based on incidental text; it is invalid. Attempts 02–03 did not
include the final JSON metadata and whole-worktree filename coverage. Use only
attempt 04 as the current screening receipt.

## Limits and next gate

- The 405 IDs are not yet a new train/validation split and are not certified
  untouched at normalized-content or media-hash level.
- The task would add generated sound to silent synthetic videos. It can test
  controlled temporal alignment and modality necessity, not natural AV
  understanding or performance on real-world audio.
- Before any feature extraction or fit, freeze a video-grouped train and
  development split, generate one deterministic timing task, and test text-,
  audio-, and video-only controls. The expected answer must require both
  modalities; paired audio timing variants for one scene must stay in the same
  split.
- Validate the archive members and each selected video's SHA-256 before
  decoding. Keep the official validation/test and all sealed-audit data closed.
- No model, feature cache, checkpoint, frozen corpus, or Teacher was changed;
  no GPU or HotUGPU job was launched. This audit used CPU metadata processing
  and 7.44 MB of annotation archive range reads.


# BSD10k v1.2 source and split audit (2026-11-28)

## Decision

Admit only the exact per-file CC BY 4.0 and CC0 1.0 subset for a narrow,
audio-only frozen-feature Path A investigation. This is a development source,
not a blind benchmark and not evidence of general Omni Decision quality. The
source is useful because it has manually assigned labels across 23 broad sound
classes, annotation confidence, and uploader provenance. Those strengths do
not remove label ambiguity, source imbalance, or pretraining-overlap questions.

## Pinned source and rights

- Dataset: BSD10k v1.2, Zenodo record `19868804`, DOI
  [`10.5281/zenodo.19868804`](https://doi.org/10.5281/zenodo.19868804).
- Source repository: [`allholy/BSD10k`](https://github.com/allholy/BSD10k).
- Task context: [DCASE 2026 Heterogeneous Audio Classification](https://dcase.community/challenge2026/task-heterogeneous-audio-classification).
- Metadata archive size: 2,726,927 bytes; SHA-256
  `c090b97e5cdac221f867d5a970b0a8326833e57cfe586c5886494170b25413f1`.
- The upstream README explains that each Freesound recording retains its own
  uploader-selected license. We fail closed to only `CC-BY-4.0` and `CC0-1.0`.
  The screen found 3,814 CC BY 4.0 and 3,334 CC0 records. We exclude 2,156
  CC BY 3.0, 1,240 CC BY-NC 3.0/4.0, and 412 CC Sampling+ records under the
  repository's source allowlist. The enclosing dataset's general CC BY label
  does not override the licenses of individual recordings.
- No DCASE 2026 evaluation media or labels were accessed. No precomputed CLAP
  embeddings or descriptive metadata are used as model inputs. The selected
  pretrained encoder is AudioSet-trained EfficientAT DyMN-0.4; overlap between
  its upstream training media and these Freesound assets is not established,
  so results must retain that exposure caveat.

## Metadata quality and split

The metadata-only screen found 10,956 unique sound IDs and 23 classes. Among
the 7,148 allowed recordings, confidence counts are 59 / 498 / 2,210 / 3,961 /
420 for scores 1 through 5. We retain all scores and will report metrics by
confidence rather than filtering difficult or uncertain labels. The original
paper/challenge description characterizes these as manual annotations; the
available metadata does not establish independent multi-rater consensus.

The deterministic seed-17 split groups by normalized Freesound uploader and
keeps each uploader wholly in one partition. Group targets are 70/15/15; the
realized counts are:

| Split | Examples | Uploader groups |
|---|---:|---:|
| Train | 4,214 | 1,185 |
| Validation | 1,436 | 254 |
| Test | 1,498 | 255 |

Uploader-group overlap is zero. The example proportions differ from the group
targets because uploader activity is highly skewed: the largest allowed group
has 426 clips. Per-class independent group support is also uneven; a few
validation classes have only three uploader groups. All reported per-class
metrics therefore need counts and uncertainty intervals, and should not be
read as equally precise. The split hash is recorded in the machine-readable
manifest. Exact media-content duplication is still pending payload-level
hash checks across partitions.

This is a custom public-development split. The metadata and labels are public,
and BSD10k has already appeared in published DCASE work. It is not a new sealed
audit and cannot support a blind generalization claim. A separate custom test
partition, if eventually used, will remain unopened until encoder/readout,
preprocessing, and calibration are frozen; DCASE's own evaluation set remains
untouched.

## Selective media access

The official Zenodo audio archive is 8,576,285,410 bytes. A range-only
central-directory inspection read 1,090,920 metadata bytes and identified all
10,956 archive entries. Exactly the 7,148 per-file allowlisted members total
5,865,317,268 compressed bytes and 7,688,893,788 uncompressed bytes. No
disallowed media payload was read during this inspection. The selector checks
the pinned metadata hash and per-file license, requests only each selected
member's local header and compressed payload ranges, and verifies the declared
size and ZIP CRC before atomically writing a WAV. It does not fetch the complete
mixed-license archive.

A 25-file train smoke downloaded 11,097,420 bytes. All 25 files passed ZIP
CRC/size checks and Python WAV parsing; the samples were mono 44.1 kHz,
16-bit PCM, with durations from 0.222 to 30 seconds. Two further batches of 25
files completed in 95.45 and 95.00 seconds respectively. This small sample
implies a rough multi-hour full train/validation transfer; throughput is
variable and the estimate is not a guarantee. The first full-train attempt was
stopped before writing files when a RAM check revealed the downloader retained
all compressed payloads until the split finished. The implementation now
decodes and writes each bounded response batch immediately. Full train
acquisition has restarted in a separate external artifact directory.
Validation is not yet downloaded; the test split and its labels are not being
used during tuning.

## Next quality gates

1. Finish and hash the eligible train media; acquire validation separately.
2. Compare source uploader IDs, split IDs, WAV hashes, and normalized PCM hashes
   across train and validation before feature extraction. Reject any overlap.
3. Extract label-free features with the pinned frozen EfficientAT DyMN-0.4
   checkpoint; do not use the bundled CLAP feature archive.
4. Train only a small fixed-taxonomy Decision readout on train features and
   select it on validation. Report accuracy, macro/per-class metrics, NLL,
   Brier, ECE, confidence strata, and uploader-group uncertainty.
5. If the candidate is frozen and useful, perform one custom test evaluation;
   otherwise record the negative result and leave test unopened.

This source screen is an R2 data admission step only. It does not close R1/R2,
establish a 95% audio task result, justify BERT changes, or validate image,
video, sensor, or joint-modal performance.


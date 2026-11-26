# Native AV command layout: fixed IDs, acquisition pending

Extends [admission](R2_LOMBARD_GRID_ADMISSION_20261013.md) with actual audio/front
ZIP central-directory and label checks. No model quality was inspected.

The two archives each contain5,390 unique, matching media stems. Speaker51 has
90files; other speakers have100. This explains the ten-asset difference from the
nominal5,400 release. Metadata has50 fewer records for speaker31, plus one unusual
repeated-recording filename for speaker14; these facts are retained, not repaired
by inventing media. Source alignmentZIP contains5,361 phoneme-alignment files,
so it is not complete independent ground truth for all recordings.

The official web page has no revision/ETag, but the actual media archives do:

- Audio652,614,041bytes,ETag`"26e61999-568649ce7ab80"`.
- Front837,239,327bytes,ETag`"31e7421f-56864af341e40"`.
- Their index SHAs are in the JSON/source pins. Only1,121,754bytes were transferred
  for the two complete directories, with exact206 ranges and stable before/after
  HEAD ETags/sizes. Full archive hashes and media CRC/SHA checks remain pending.

## Strict labels and connected splitting

Native tasks are the six spoken-command slots:command,color,preposition,letter,
number,adverb. Six-character codes are parsed against exact upstream grammar;
`WRONG` uses the actual speech suffix, never the intended prompt. Free-form
error/silence/distortion markers remain unresolved. Do not strip markers to make
a plausible label. Metadata and filenames disagree for one otherwise-valid
record:quarantine it rather than choose the convenient label. Filename-only labels
are explicit where metadata is absent; their full-media validation remains pending.
The single exceptional repeated recording retains its distinct source identity.

5,371labeled/5,390published assets (99.6475% label coverage);19 retained unresolved:
18unparseable actual annotations and1valid-label conflict. Coverage is not Accuracy;
the unresolved records cannot silently become correct answers or disappear from
the denominator when describing source coverage.

Seed17 joins all assets from one speaker and all speakers sharing valid intended
or actual complete phrases; includes published6/29 and25/26 aliases.49 connected
groups remain. SHA256 of seed+sorted group assigns <70train,70–84calibration,
85–99validation buckets. No seed or bucket search was performed.

| Partition | Labeled assets | Speakers | Lombard/plain |
| --- | ---: | ---: | ---: |
| Train | 4,084 | 41 | 2,039/2,045 |
| Calibration | 398 | 4 | 198/200 |
| Development validation | 889 | 9 | 450/439 |

Zero known source/speaker/complete-phrase overlap. Unresolved labels still receive
the same speaker-group split; no individual recording is reassigned to improve
balance. Options are fixed slot vocabularies, with no label reduction. Small
calibration/validation speaker counts limit uncertainty; this is not a blind audit.

## Next execution gates

Corpus JSON IDs/labels/layout and hashes are fixed in
`C:/CodexArtifacts/pretrained-reuse-r2/lombard-grid-layout-01/`.
This does **not** certify media acquisition or readiness for learning. Require
fixed-ETag archive acquisition, all expected member CRC/size/SHA checks, decoding,
cross-split duplicate rejection and an explicit AV-timing profile first.
No encoder training or new model metrics have been run.

Inputs must be waveform and video pixels/timestamps only. Filenames, source IDs,
metadata, intended/actual transcripts and gold alignments are provenance/labels,
never features. Any future inference uses native option probabilities/JSON without
requiring a question. This is a scoped speech-command capability; it cannot replace
the image/environmental-audio/temporal-video task battery or certify world reasoning.
Joint benefit must be measured against unimodal and mismatched-evidence controls
without contrived degradation. Existing artifacts are preserved; paid compute0.

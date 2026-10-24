# Perception Test audio-to-visual source candidate preflight

## Status

This is a train-only task-format, annotation, and limited media-integrity result.
It does not admit Perception Test as a product-quality benchmark, establish that
both modalities are required, or report model quality. No official validation
or audit annotations were loaded. Thirty-two official-train media files were
decoded for CPU integrity checks; no model was loaded or fitted.

## Source and reproducibility

- Dataset revision: `3938d2f1ba3a6b502025741cea4cd73c7b3bdfaf`.
- Inspected sources: official `all_train.json` and train cut-frame map only.
- Train annotation archive SHA-256:
  `9eaa6aad3fabdf4cde4955f333cfddf3b9e6839c5abcb271afa410870f640d78`.
- Train cut-map path:
  `C:\CodexArtifacts\pretrained-reuse-r2\perception-test-source-audit-01\cut_frame_mapping_train.json`.
- Cut-map SHA-256:
  `cbea6c721fd150f861ec6fe8e7ff005792f754df1359b1364349b5abcaab6cc4`.

The cut map's 2,184 video IDs must match the train annotations exactly (plus
the upstream `nid` entry). Generated annotation-only corpus and split data are
stored outside Git:

- Corpus directory:
  `C:\CodexArtifacts\pretrained-reuse-r2\perception-source-candidate-preflight-20261025-06\`.
- Corpus manifest SHA-256:
  `91322da5508977cdbd2f693819836b7496ed484ea55b7eb193d71e54e1e4618d`.
- Corpus JSONL SHA-256:
  `f3f51926c15658b05aa27ca548001f9f4cb6232b1a5cfbb1e6bcbfd706538701`.
- Adapter SHA-256:
  `b1c7a6b047dbc39409934ecf663c0ffc92e397b5895f85f8adc576c8eb787f8d`.
- Corpus builder SHA-256:
  `6e330e96d87e61d9a4dec9c99bdb3a26499dde8903b6196f8641169a37ae20db`.

## Task definition

Given an audio interval and object-crop proposals from overlapping video frames,
select every visible object track annotated as a source of that sound. If the
source is annotated invisible, select the explicit `none_visible` option. The
pinned official README describes `is_visible` as whether the objects causing the
sound are visible; `parent_objects` may be empty, multiple, or `-1` when
unannotated. Accordingly, invisible events are retained as explicit
none-visible supervision. Visible events with empty or unknown parents are held,
not converted to negatives.

This is **candidate re-ranking conditional on an oracle event interval and
oracle object tracks**. It does not detect event boundaries, discover object
proposals, or provide an end-to-end real-time audiovisual decision. The source
semantics still need review before interpreting a model result as causal source
identification.

All annotated parent objects remain positive; multi-source events are not
collapsed to one arbitrary target. A visible event is admitted only when each
source parent has an unmasked box in the interval and at least one non-parent
object is also present. An invisible event requires at least two visual
candidates, avoiding an empty-candidate shortcut. Every event must end before
audio duration, video duration, and the upstream answer-reveal cut. Masked
boxes, unknown visible parents, missing parents, no-distractor visible events,
trivial invisible events, and boundary-crossing events are held with reasons.

Runtime request fields exclude sound labels, object labels, annotation IDs,
video IDs, and source object IDs. Candidate IDs use deterministic,
per-event hash ordering (`option_N`). Media identity, grouping, cut provenance,
and targets remain outside model-request fields. Downstream collators must not
serialize `media_reference`, `group_id`, `provenance`, or `supervision` as model
features. Up to four evenly spaced visible boxes per track are retained; this
sampling policy has not been validated. No crops or model tensors are in the
corpus.

## First-generation cut-aware annotation yield (superseded)

The pinned adapter processed all 25,757 sound intervals across 2,184 train
videos:

| Status | Events |
|---|---:|
| Candidate | 14,741 |
| Held | 11,016 |

Candidates span 2,132 videos. Of 14,741 candidates, 4,788 use `none_visible`;
the others preserve one or more positive source tracks. Candidate sets contain
2–49 objects. The full candidate-count, positive-count, source-label metadata,
and hold-reason distributions are in the external corpus manifest. Across the
source annotations, 247 event intervals exceed audio duration and none exceed
video duration; additional events are held by cut-frame rules. These are
annotation counts, not accuracy results.

This first-generation corpus replaced the earlier narrower exploratory subset,
which omitted invisible events and did not enforce the pinned cut map. It was
subsequently found to contain repeated inference inputs and label conflicts;
it is preserved for provenance and must not be used for training. The held
visible events with unknown parent IDs remain unknown rather than becoming
`none_visible`.

## Local train media and CPU smoke

Read-only filename matching against the existing
`perception-av-event-train-media-01` acquisition found 400 local official-train
videos, covering 3,219 current candidate events, including 881 none-visible
events. This is file presence only.

A seeded CPU smoke selected 32 candidate events from 32 distinct train videos.
It required at least eight none-visible events and at least one event from each
of the four local groups with an answer-reveal cut; the final sample contains
14 none-visible events and four cut groups. All 32 MP4 sizes and SHA-256 values
matched the acquisition receipt. PyAV decoded all 107 requested bbox frame
IDs; presentation times were within the event interval using a two-frame
tolerance. Decoded audio covered 106.552765 seconds across all events. This
checks 32 of 400 files and does not decode candidate crops or calculate model
metrics.

- Report:
  `C:\CodexArtifacts\pretrained-reuse-r2\perception-source-candidate-media-smoke-20261025-05\media-smoke.json`.
- Report SHA-256:
  `3a4336ee1f09ff7dcd6a37f2109ccb0343bf69966078d95141153bf810fb085d`.
- Media smoke script SHA-256:
  `9a2969bd288d69a471a74bceecbbfc63c69398b1ea3462345a335997f2034d44`.

### Full local-media cohort check (2026-10-25)

A second seeded CPU preflight sampled one candidate event from each of the 400
locally available candidate-bearing official-train videos. It required at least
eight none-visible cases and all four available cut-frame groups. All selected
media hashes matched the acquisition receipt; 1,264 of 1,264 requested bbox
frames decoded, and 1,438.766637 seconds of requested event audio had complete
decoded coverage. The report records 122 none-visible examples and four
cut-frame videos. No model was loaded, no metrics were computed, and no GPU was
used. This checks one event per video, not every annotation: the other 2,819
candidate events in these local videos remain unchecked.

- Report:
  `C:\CodexArtifacts\pretrained-reuse-r2\perception-source-candidate-media-preflight-attempt-06\media-preflight.json`.
- Report SHA-256:
  `a0d171258361a55531ee6acf92306c54809bb0d87ad61507c760c41e54c39230`.
- Command used `scripts/preflight_perception_source_candidate_media.py`
  with seed 17, limit 400, eight required none-visible cases, and all four
  required cut-frame groups.

### Full frozen development-record check (2026-10-25)

The explicit-record path checked all 611 records from the frozen internal
development JSONL across 80 video groups, preserving the JSONL record order.
All 1,927 requested bbox frames decoded, and 2,026.274091 seconds of event
audio had complete decoded coverage. All 80 local media hashes matched; 176
records were none-visible. This cohort has no cut-frame videos. It is a
non-blind, train-source internal development partition, not official
validation or sealed audit. The 611 records overlap 80 records in the earlier
400-video sample, so 931 distinct records have been checked across both
reports. Of 3,219 candidate events in the 400 local videos, 2,288 remain
unchecked.

- Report:
  `C:\CodexArtifacts\pretrained-reuse-r2\perception-source-candidate-dev-media-preflight-attempt-07\media-preflight.json`.
- Report SHA-256:
  `522ec41de11bc3c23aeba7172e4e3a8614415a357d987cc513461a0c5c571075`.
- Selection source is the frozen development JSONL, SHA-256
  `51fc3ee12471e6d884a8808e5d84fce42d4d3de0214b325176bc6d374b455f89`.
  In this report the field name `selected_record_ids_sha256` refers to the
  source JSONL file hash, not a separately serialized ID-only list.
- The executed media checker was based on source commit
  `d0fd6865ce2faf83bfb840cbd3bd96081350eb45`; its executed script SHA-256 was
  `32fee67b9ea122dd8ff355980ea3febf0624ac71c83c30910ee3c1b8600a4e7e`.

## Provisional development split

A seed-17 deterministic hash split was frozen over the 400 candidate-bearing
official-train videos with local media: 320 train groups / 2,608 events and 80
development groups / 611 events. Video-ID overlap and exact media SHA-256
overlap are both zero. This split is not blind: prior Perception Test work used
the same video media for other labels, and train annotation statistics have
been inspected. Participant-level identity is unavailable, and normalized
near-duplicate content has not been checked. This must not be reported as a
clean independent validation or sealed audit.

- Split directory:
  `C:\CodexArtifacts\pretrained-reuse-r2\perception-source-candidate-dev-split-20261025-02\`.
- Manifest SHA-256:
  `dd460e41f500e9900b47512ad30c01a41369f5d631efe163fc096448689f97f3`.
- Train JSONL SHA-256:
  `a132536a824446f8b166162f2daefe2011570dee3a3f7900073c1232c9c34d26`.
- Development JSONL SHA-256:
  `51fc3ee12471e6d884a8808e5d84fce42d4d3de0214b325176bc6d374b455f89`.
- Splitter SHA-256:
  `a371655d00ca66b245dff91f099e592a117ff22ae7e888457fa9373738e8ea3b`.

The 1,732 other candidate-bearing train groups without local media are not in
this pilot and were not assigned to either partition.

## Corrected fixed development split and label-blind controls (2026-10-25)

After the corrected corpus identity audit, the seed-17 split was regenerated
from only the 400 locally acquired official-train video groups. It contains 320
train groups / 2,593 examples and 80 internal development groups / 606
examples. Video and exact-media-hash overlap are zero. The development group
is a non-blind internal split because the official-train source and other
labels from these videos have been examined; it is not a new validation or
audit set.

A label-blind, deterministic intervention manifest was frozen before any
feature extraction or fit. It maps every fixed example ID to a different-video
audio donor within its own partition, a one-second same-video audio shift when
the complete shifted interval remains inside the audio/video/reveal-cut bounds,
a deterministic candidate permutation, first-frame-only candidate views, and
reversed candidate-frame order. It also names audio-only and video-only
ablations. Donor mappings are group derangements, so no donor video equals the
recipient video. Candidate IDs are retained so predictions can be joined back
without changing the answer key. The manifest contains no labels, model
predictions, or scores and records no model load or metric computation.

The frozen manifest covers all 2,593 train and 606 development rows. A bounded
one-second offset is available for 2,431/2,593 train rows and 570/606
development rows; other rows are retained in the primary condition and marked
ineligible for this particular offset intervention. The candidate ordering is
non-identity for every row (2,593/2,593 train; 606/606 development). Frame
reversal is informative for 1,179 train and 271 development examples with at
least one multi-frame candidate; other rows remain in the main conditions. The
cross-video shuffle tests dependence on synchronized audio but does not
guarantee semantic
contradiction. No contradiction control, quality gate, or task-semantic
validation is claimed yet. The next model experiment remains blocked on full
corrected-train CPU media preflight, feature/input parity, and explicit source
label review. No model has been loaded or fitted.

Fixed-split structure is uneven and must be represented explicitly in any
readout: train has 692/2,593 `none_visible` rows and 1,901 visible-source rows
with one to six positive candidates; development has 172/606 `none_visible`
rows and 434 visible-source rows. Mean candidate counts are 14.70 (train) and
14.45 (development), with a range of 2–35. The source audio interval duration
is not a fixed model window: the train median is 1.002 seconds (range
0.067–35.029 seconds; 225 intervals exceed 10 seconds), and the development
median is 0.935 seconds (range 0.099–35.017 seconds; 49 exceed 10 seconds).
Any EfficientAT input adaptation therefore needs a fixed, documented
short-interval padding and long-interval multiwindow policy, applied identically
to train and development. The existing fixed two-second extractor cannot be
silently reused as if these durations were already two seconds.

- Corrected split directory:
  `C:\CodexArtifacts\pretrained-reuse-r2\perception-source-candidate-dev-split-20261025-03\`.
- Train JSONL SHA-256:
  `c5a8396a6b47a8635033e445c67233c76a6ecb17d27f4e6943a153fd3598600d`.
- Development JSONL SHA-256:
  `701bd1ab203e54fae5fcb7cc8e5987c2d903bd23b5d2fbf736c2132e953ebcda`.
- Split manifest SHA-256:
  `28b0bf9177d1ff2d2b89adb0c7d2916fc01180aef1d51da5b65356849ffd53aa`.
- Control manifest:
  `C:\CodexArtifacts\pretrained-reuse-r2\perception-source-candidate-controls-20261025-03\control-manifest.json`.
- Control manifest SHA-256:
  `5f7ee054a31eceafd6a21855e3d4c4569c93c50ee2a51ccaee472c8f1268d93`.
- Control manifest generator SHA-256:
  `df33a54b66869a62678e54fc6f918fa9853c5d7830e0a13ccd907c26541d414c`.
- Control generator tests SHA-256:
  `5cc73aaede16ec33ff6bdfef9c19d03fa915c3a01c1987f0847c8057cd7c4244c`.
- Targeted result: 5 control-manifest tests passed; Ruff passes for the
  generator and its tests.

The full corrected train media check is running on CPU against all 2,593 fixed
train rows. At this report update, it had decoded through row 640; this is an
in-progress check, not a pass. The development media check remains the prior
complete 606-row report above. GPU work and learning remain gated on a terminal
train-media result and the remaining protocol gates.

## Input identity audit and corrected corpus

The first-generation corpus was audited before any model training. Its 14,741
candidate annotations contained 71 repeated model-input groups (146 records),
including two contradictory groups (four records). The input signature ignored
event-specific candidate IDs and option order and compared exact video/time
identity plus hashes of each normalized candidate frame-box sequence. It also
found 32 records where a positive and distractor candidate had identical frame
box sequences, and two candidate intervals coincided with unresolved held
annotations. The source annotation audit showed the contradictory cases paired
a visible source with an invisible-source annotation over the same interval.
No sound class or event identity was provided to inference, so those per-event
targets could not be distinguished by the model input.

The builder now applies fixed identity gates before model use:

- coalesce 73 repeated rows from 69 groups when the exact model input and target
  agree, retaining their annotation IDs as provenance;
- hold all four rows in the two contradictory groups;
- hold two otherwise-candidate rows that share an interval with a different
  unresolved held annotation;
- hold 32 rows where a positive candidate crop is visually indistinguishable
  from a distractor under the frozen box input.

All 25,757 source annotations remain represented as candidate, held, or
duplicate provenance rows. The corrected corpus has 14,630 candidate examples,
11,054 held examples, and 73 duplicate markers. A second identity audit found
14,630 unique model inputs, zero repeated groups, zero contradictory groups,
and zero partially targeted identical-crop rows. This is a predeclared input
integrity gate, not performance-based example filtering. Cross-video
pixel-level duplicate crops and the annotation's source-label semantics remain
unverified.

- Corrected corpus directory:
  `C:\CodexArtifacts\pretrained-reuse-r2\perception-source-candidate-preflight-20261025-07\`.
- Corpus JSONL SHA-256:
  `11cd4c758b04e36702969d168a9987dde8028d75733bb95a0a684d7f712a19d6`.
- Corpus manifest SHA-256:
  `655713c9f1dd61970081b28d0ebbc0f7ddb001b8ccea68576c237b8239d21430`.
- Pre-consolidation identity audit report SHA-256:
  `0ac3a82b64553fde0f77d49ac91b675062665d5cc75c213d51fce8c1c8fc9def`.
- Post-consolidation identity audit report SHA-256:
  `67d51826d373370f19cbf5a99bd9c541ee0cfbbcc969489364dad8e093784ab5`.
- Identity auditor SHA-256:
  `971feac98efd727a38bf867dc3fcae08e15ac105219430c27bd48aae12380682`.
- Adapter SHA-256:
  `b231b269b55614ef8d1b1a308472dbbcea0442eac43148bee2396e23aaecdd34`.
- Corpus builder SHA-256:
  `36288d2e7aa0f0bd07702b3925aaf280ee35743625c11eb83eaf3d975e27e7e4`.

A replacement seed-17 split was frozen from the corrected corpus, with the
same 320/80 video groups but 2,593 train and 606 development records. Video ID
and exact media SHA overlap remain zero. The development JSONL is still
non-blind and inherited from official-train videos; its SHA-256 is
`701bd1ab203e54fae5fcb7cc8e5987c2d903bd23b5d2fbf736c2132e953ebcda`.

All 606 corrected development records were then decoded on CPU. The check
covered 1,840/1,840 requested frames and 1,921.820085 seconds of event audio,
with 80/80 media hashes matching the acquisition receipt and 172 none-visible
records. It loaded no model and computed no quality metrics.

- Replacement split manifest SHA-256:
  `28b0bf9177d1ff2d2b89adb0c7d2916fc01180aef1d51da5b65356849ffd53aa`.
- Replacement train JSONL SHA-256:
  `c5a8396a6b47a8635033e445c67233c76a6ecb17d27f4e6943a153fd3598600d`.
- Development media report:
  `C:\CodexArtifacts\pretrained-reuse-r2\perception-source-candidate-dev-media-preflight-attempt-08\media-preflight.json`.
- Development media report SHA-256:
  `a18a0fe1c346f20165cd5f1fbfaa993eb68616edc535bee2d5b35332bda12c94`.
- Development media checker SHA-256:
  `ac8a1088de6b855006546c42afc988710ffe084773e898d49518f8c124e7bf8e`.

## Remaining gates before model work

1. Every record in the 80-group internal development partition was checked.
   The corrected 320-group train partition has 2,593 candidate records; a
   current-corpus train media check is still required before feature extraction
   or fitting. There is no local media for the other 1,731 candidate-bearing
   groups in the corrected corpus. Do not load previously inspected official
   validation or any sealed-audit annotations.
2. Freeze matched audio-only, video-only, event-label-prior, candidate-order,
   audio-shuffle, stale-window, and interval-offset controls before selecting a
   model. Match candidate sets and report multi-label and none-visible metrics.
3. Check whether the task reduces to sound-class or object-order priors, and
   verify the annotation's parent-object semantics for source matching.
4. Report participant-grouping and near-duplicate limitations. A video-grouped
   split alone does not establish participant independence.
5. Continue using only media admitted for this train-only purpose. Do not start
   GPU work until the controls and data checks are fixed and the HotUGPU guard
   permits launch.

No model training, validation scoring, sealed-audit access, quantization, or
product promotion has occurred. The RTX 3080 remained occupied by Orbitune, so
all work in this preflight used CPU.

## Implementation

- Adapter: `src/tiny_omni_decision/perception_source_candidate.py`.
- Deterministic corpus writer: `scripts/build_perception_source_candidate_corpus.py`.
- Group split freezer: `scripts/freeze_perception_source_candidate_split.py`.
- CPU media preflight: `scripts/preflight_perception_source_candidate_media.py`.
- Tests: `tests/test_perception_source_candidate.py`,
  `tests/test_freeze_perception_source_candidate_split.py`, and
  `tests/test_perception_source_candidate_media.py`.
- Existing visibility-proposal format remains separate and unchanged.

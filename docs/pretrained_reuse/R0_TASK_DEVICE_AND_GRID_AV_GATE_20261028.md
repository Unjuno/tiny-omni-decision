# R0 task/device targets and GRID synchronized AV pilot

Status: criteria and the pre-media sample are frozen. A source folder-mapping
integrity audit blocked AV evaluation before inference; this document contains
no new quality result and does not approve GRID for product distribution.

## Product target and device

- The model returns a JSON object with normalized probabilities for the
  supplied choices, preserving choice identity and input order. It does not
  need general-purpose text generation.
- Count every resident encoder, tokenizer embedding, lens, fusion/readout and
  decision head in the pre-quantization parameter budget. The product target is
  100–200M total parameters; the roadmap's <=200 MB deployment-package target
  applies only after a measured quantized export and functional parity check.
- Primary engineering device is the local Windows PC with the NVIDIA RTX 3080
  Laptop GPU (16 GiB). CPU execution is the fallback for bounded feature
  extraction and tests. No cloud compute is in scope.
- The user's 95–98% goal is recorded as an aspirational accuracy range on each
  defined task separately. It is not an aggregate-across-modalities pass and
  must not be asserted from the small GRID pilot. Report Accuracy, NLL, Brier,
  ECE, calibration bins, independent groups and confidence intervals per task.
- Sensor input remains optional and outside the first acceptance suite because
  there is no approved sensor task paired with a decision label in the current
  inventory.

## First task suite

The suite is intentionally task-specific; success on one row does not imply
general modality understanding.

| Input | First measurable task | Interpretation limit |
|---|---|---|
| Text | Conditional choice selection on the existing Typed Decisions Synth and eligible Open-Jev records | Synthetic examples measure rule following; source/state groups must not cross splits. |
| Image | CLEVR-4 supplied-option attribute/taxonomy decisions | Synthetic visual classification, not natural-image understanding. |
| Audio | Speech Commands v0.02 ten-way spoken-keyword choice, with the separately frozen LibriSpeech hard-negative evaluation when available | Does not establish general speech comprehension from keyword accuracy. |
| Video | CLEVRER supplied-option temporal decisions, with descriptive and temporal question types reported separately | Group by video scene; no claim beyond measured question types. |
| Synchronized audio+video | GRID four-choice utterance-to-video correspondence, defined below | Narrow speech-perception correspondence only; not general audiovisual reasoning. |

For every row, preserve the existing source/group split rules and report both
unimodal baselines and the full input path. Never tune on sealed audit data.
The 95% target remains a task-by-task objective; the 98% value is a stretch
goal, not a substitute for calibration, OOD and independent-group evidence.

## Frozen GRID development pilot

The task is: given one audio utterance and four synchronized candidate video
clips, return a JSON probability for each clip and select the clip from which
the utterance was recorded. The system receives no speaker ID, transcript,
alignment code, quartet ID or target. The evaluator retains those labels.

- Start from the pinned GRID two-slot correspondence validation manifest and
  its fixed 24-speaker train / 8-speaker validation split. Keep all four query
  roles per quartet so each candidate is the correct answer once and candidate
  position cannot determine the label.
- Before reading validation media or model outputs, select the eight quartets
  per validation speaker with the lexicographically smallest
  `SHA256("grid-av-pilot-v1|" + quartet_id)`. This gives 64 quartets, 256
  queries and at most 256 distinct candidate videos before cache reuse. Keep
  the eight speaker clusters as the independent units; report uncertainty as
  low-power and do not claim a 95% threshold from this pilot.
- Compute the matched audio+video result and audio-only, video-only, permuted
  option, mismatched-audio and mismatched-video controls on identical queries.
  The quartet role balance is required for interpreting position controls.
- Use the frozen model, preprocessing, temperatures and scoring code from the
  existing Lombard-trained correspondence runtime. No GRID training, threshold
  fitting or checkpoint selection occurs in this pilot. Record all per-query
  probabilities and pair them by quartet for controls.
- The provisional speaker split prevents GRID speaker-label overlap, but
  identity overlap between GRID and the model's Lombard training speakers has
  not been verified. Therefore the pilot cannot claim cross-corpus person
  independence or serve as a sealed product audit.

## Rights and distribution boundary

The official [Zenodo v1.0 record](https://zenodo.org/records/3625687) contains
34 talkers and lists `cc-by-4.0` in its record API metadata. The official
[University of Sheffield corpus page](https://spandh.dcs.shef.ac.uk/gridcorpus/)
describes the corpus as freely available for research use. These statements
are retained together; this project does not interpret the research-use notice
as product or redistribution clearance. Use any downloaded files only for
internal research evaluation, preserve attribution, do not commit or package
the media, and keep GRID out of product artifacts pending rights clarification.

The raw Zenodo API response was saved outside Git at
`C:/CodexArtifacts/pretrained-reuse-r2/grid-validation-video-20261028-attempt-01/zenodo-record-3625687.json`
(SHA-256 `fbcf06eecb05bdb8cb93a4b1cd6d1d4bfad81f5a1545e008c137d23bf1265c7e`).
It reports `cc-by-4.0`, open access and the publisher's per-file MD5 values.
The first held-out speaker archive `s30.zip` completed at 381,061,085 bytes;
publisher MD5 `8958f25b49de7a7827a76abbdc7fedb2` matched and its SHA-256 is
`69e7c2ef14b1bd4f2d0012a8a13fe8e9375c4f11a03c4de9f03f4071f48f12fd`.
One additional archive download was stopped after the speaker-folder mismatch
was found; its incomplete `s4.zip.part` is preserved but unverified. The other
six held-out archives were not downloaded. Do not resume these downloads until
the source mapping is resolved.

## Evidence and remaining gates

- Pinned split, two-slot catalog and hashes: see
  `R2_GRID_DEVELOPMENT_SPLIT_PROTOCOL_20261028.md` and
  `R2_GRID_TWO_SLOT_CORRESPONDENCE_PROTOCOL_20261028.md`.
- One fixed same-speaker JSON interface smoke selected the wrong clip; it was
  a technical smoke, not a quality evaluation. See
  `R2_GRID_MATCHED_AV_JSON_SMOKE_RESULTS_20261028.md`.
- The present pilot requires verified per-speaker archive hashes, media/code
  matching, exact sample-list hash, runtime parity, all controls and saved
  probabilities before reporting any task metric. Product rights, cross-corpus
  speaker identity, broader task validity, all modality quality targets and
  the 100–200M / <=200 MB device budget remain open.

### Frozen sample selection receipt

The selection script and focused tests were run against the pinned validation
manifest before reading validation media or predictions. The external receipt
is `C:/CodexArtifacts/pretrained-reuse-r2/grid-av-pilot-selection-20261028-attempt-01/summary.json`.
Selected `selected.jsonl` SHA-256:
`b33e4e7e2cdb77ffdfa58da371fd712dd57c63341651cfb416505adf57ccbf0d`.
The selection contains 64 quartets / 256 balanced query roles from eight GRID
speaker labels, with 249 speaker-qualified audio/video codes. The source manifest SHA
is `ff7ec51c8bf8e873c326d41771a822b400ec99a3a8d5ee6871ad92647a364aa2`.
No media, model predictions or sealed audit were read during selection.

### Media integrity gate failed before inference

The exact official GRID audio ZIP passed its publisher MD5 and SHA-256, but
the pinned alignment speaker labels fail same-name audio-folder coverage for
18,997 of 31,991 aligned speaker/code rows. Four of the eight validation
speaker labels in this pilot map all their codes to a different audio folder
by filename overlap. That overlap does not establish person identity. No audio
or video prediction was run; do not use this frozen sample until the source
provider explains the mapping or an authoritative mapping artifact is found.
The full read-only audit is in
`R2_GRID_AUDIO_FOLDER_MAPPING_AUDIT_20261028.md`. The sample list and its hash
remain unchanged.

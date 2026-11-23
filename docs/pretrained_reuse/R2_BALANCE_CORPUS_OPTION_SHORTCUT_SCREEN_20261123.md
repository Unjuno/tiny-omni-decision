# Balance Corpus option-shortcut screen — 2026-11-23

## Disposition

**Do not admit the released Taboo target-selection task as R2-M fusion-quality
evidence.** The latest official Zenodo version is open under CC BY 4.0 and
contains 1,200 trial rows from 30 dyads. Its target and five taboo words form
40 unique unordered option sets. Every option set is reused across pair groups
and maps to the same target. A split by dyad therefore leaves an exact
option-set-to-answer shortcut across train and validation. The 30 dyads also
fall well below the roadmap's provisional 200 independent-group floor.

This conclusion comes from metadata only. No audio/video, demographics, or
gyroscope media was downloaded or opened; no model, feature extractor, GPU, or
training run was used. The source is useful as a warning about task design, not
as an admitted candidate for multimodal quality measurement.

## Source and metadata integrity

- Official latest record: [Zenodo 21648711](https://zenodo.org/records/21648711),
  DOI `10.5281/zenodo.21648711`; the Zenodo API reports `open` and `cc-by-4.0`.
- The record lists 1,200 Taboo trials from 30 participant pairs, with audio,
  four camera views per trial, gyroscope data, transcripts, and trial metadata.
  The complete media bundle is 129.6 GB, chiefly three video archives totaling
  about 129 GB. No media was fetched.
- Retrieved only `metadata.csv` (527,631 bytes) from the official record.
  SHA-256: `9c8dd4c97fb7f8d8ea6bb446fc41038e7066c667952e7d8a7b06fa67f0b0ca6a`.
- The metadata contains 40 trials per pair, 600 `board` and 600 `ground`
  conditions. Every row has six distinct candidate words.
- The machine-readable report intentionally omits participant/pair identifiers
  and all raw text. It is held outside Git at
  `C:\CodexArtifacts\pretrained-reuse-r2-local\balance-corpus-source-screen-20261123-attempt-01\`.

The peer-reviewed Warwick paper reports informed consent and describes the
clue-giver as receiving the target plus five taboo words while the guesser is
not shown them. In the released trial metadata, the unordered six-word set is
reused across dyads, and the same item is always the target. Thus an
option-only lookup can answer the released task without audio or video. This is
an exact task-label shortcut, independent of the model family or fusion design.

## Reproducible audit

```powershell
python scripts/audit_balance_corpus_option_shortcuts.py `
  --metadata-csv C:/CodexArtifacts/pretrained-reuse-r2-local/balance-corpus-source-screen-20261123-attempt-01/metadata.csv `
  --output C:/CodexArtifacts/pretrained-reuse-r2-local/balance-corpus-source-screen-20261123-attempt-01/summary.json
```

The script reports aggregate counts and the metadata hash only. It does not
emit identifiers, targets, taboo words, transcripts, or participant metadata.
Focused unit tests cover deterministic option-set leakage, conflicting labels,
and missing schema fields.

## Decision

Do not spend the 129 GB media budget on this task. A dyad-group split would
still leak the answer through the repeated option set. Grouping by option-set
leaves only 40 independent candidate sets and still does not establish that
both audio and video are necessary. A different target, such as a separately
annotated gesture interpretation task, would require new label provenance and
independent groups; it cannot be inferred from these target labels. The source
does not advance the R2-M quality gate in its current form.

## References

- [Official Zenodo record and file manifest](https://zenodo.org/records/21648711)
- [Warwick paper record](https://wrap.warwick.ac.uk/id/eprint/200814/)
- [Accepted paper PDF](https://wrap.warwick.ac.uk/id/eprint/200814/1/WRAP-perturbing-postural-stability-affects-acoustic-properties-fluency-2026.pdf)

# Clotho-AQA: train-only admission audit

Following the failed positive-only transfer experiment, inspect a task with
human yes/no answers to explicit audio questions, rather than interpret absent
FSD labels as negatives. This is candidate admission, not a model result.

Official immutable record: https://zenodo.org/records/6473207 (DOI
10.5281/zenodo.6473207). Downloaded only metadata, official train questions and
LICENSE.txt; each passed the record's MD5 and has a saved SHA-256. No validation
or test questions, media, or sealed audit were downloaded/opened. The record
metadata lists their filenames and sizes only. Questions/answers are MIT;
audio rights are per clip, not inherited from the question license.

## Measured admission

- Official train: 21,132 annotation rows, 1,174 named audio clips.
- Matched metadata: 7,037 distinct filename/question pairs; six additional
  questions have an unresolved filename mismatch. HTML entities are decoded
  collision-free for matching. A dot/underscore mismatch remains quarantined;
  no fuzzy filename substitution is authorized by this audit.
- Matched binary questions: 4,691; 2,762 unanimous and 1,929 with disagreement.
  One has six annotations instead of three and is quarantined.
- Accept only exact CC0 / CC-BY 3.0 licenses. Exclude known FSD128 training
  source IDs and uploaders. These sequential filters exclude 800 binary
  questions by license, four by source ID, and 464 more by uploader.
- Remaining candidate pool: **3,422 questions, 856 clips, 564 uploaders**;
  majority labels 1,953 yes and 1,469 no. Disagreement is retained, not used
  to select an easier unanimous benchmark. Confidence is not a selection filter.

These numbers do not prove independent evaluation. A future split must group
by uploader/source identity, preserve all questions for one audio, audit
normalized content and waveform reuse, and check other existing corpora.
EfficientAT pretraining overlap remains unverified. Multiple questions and
annotators are not independent audio samples. Human negative answers are
question-specific; they cannot be relabeled as arbitrary event negatives.

## Next experiment boundary

Use a frozen small audio encoder plus a pretrained text encoder and conditional
readout; compare against question-only and audio-only controls on identical
source-grouped development IDs. This would test dependence on both observation
and criterion. No architecture, split, budget or training is launched here.
Majority-vote and disagreement reporting must be frozen before scoring. Existing
STARSS/FSD results and product goals remain unchanged.

Local evidence: `C:/CodexArtifacts/pretrained-reuse-r2/clotho-aqa-admission-01/`:
record.json, verified input files, downloads.json, audit.py,
eligible-training-questions.json, result.json. Audit ran on CPU. The first
download used the wrong LICENSE basename; verified CSVs were reused after
resolving LICENSE.txt from the record. Console decoding failed once after
successful verification; UTF-8 reads corrected it. No failed attempt was a
model experiment. Compact result and exact audit source accompany this report.

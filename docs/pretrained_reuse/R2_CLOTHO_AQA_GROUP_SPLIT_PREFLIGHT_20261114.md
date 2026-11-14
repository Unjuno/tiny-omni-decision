# Clotho-AQA train-only grouped split preflight (2026-11-14)

## Scope

This advances the existing metadata-only [Clotho-AQA admission screen](R2_CLOTHO_AQA_ADMISSION_20261013.md).
It uses only its pinned official TRAIN-derived eligible question file. No
audio media, official validation/test questions, model weights, quality labels
from other splits, or sealed-audit content were loaded. No model was run and no
training or quality metric was produced.

The [official Zenodo record](https://zenodo.org/records/6473207) identifies the
audio archive as 3.14 GB, says each clip has its own Freesound license in the
metadata, and places the question/answer CSVs under MIT. The next acquisition
must therefore select only the already screened TRAIN clip names; the complete
archive, which includes all 1,991 recordings, must not be downloaded or opened.

The source task is binary audio-question answering with three retained human
votes per item. It is a candidate audio-plus-text Decision task; both-input
necessity is not established until matched question-only and audio-only controls
are measured on a valid development set.

## Reproducible assignment

- Input SHA-256:
  `58d6e585b310a7ebc804761f00cc412aa07aeec58411ee418f4944f9fc069420`.
- Input contains 3,422 eligible questions, 856 clip names, and 564 uploader
  groups. The existing admission filter allows only exact CC0 or CC-BY 3.0 clip
  licenses and excludes known FSD128 IDs/uploaders; the split script independently
  checks each row's license and majority label against its three votes.
- Seed 17. Uploader IDs are ordered by SHA-256 of the fixed seed/dataset/group
  string. Assignment is by group rank, without label stratification: 264 groups
  to train, 100 to calibration, 200 to development. These counts are a frozen
  feasibility proposal using the provisional group targets discussed in PR #98;
  that draft is not a final release contract.
- The full split-membership output is outside Git at
  `C:\CodexArtifacts\pretrained-reuse-r2\clotho-aqa-group-split-20261114-attempt-02`.
  Its SHA-256 is
  `a459641e5d6f1b653701dd82fe45b17c0b7b52847e1325d1ef4fc6d07857889`.
  Fresh-process replay in `...attempt-03` produced byte-identical membership.

| Split | Uploader groups | Clip names | Questions | No | Yes | Questions with vote disagreement |
|---|---:|---:|---:|---:|---:|---:|
| Train | 264 | 390 | 1,560 | 654 | 906 | 635 |
| Calibration | 100 | 153 | 610 | 275 | 335 | 259 |
| Development | 200 | 313 | 1,252 | 540 | 712 | 535 |

Uploader, source-ID, and clip-name overlap across these partitions is zero.
Because the source material has not been acquired, waveform-hash and normalized
audio-content overlap remain **unchecked**. The count of questions with vote
disagreement is retained as evidence; no unanimous-only filtering occurred.

## Important limitation

Normalized question wording does recur across partitions: 80 train/calibration,
135 train/development, and 69 calibration/development shared question strings;
39 normalized question strings occur in all three. These are different
audio-question records by source ID, so this is not an exact multimodal-record
duplicate, but it means development is not a test of unseen question wording.
The repeated-question rate must also be exposed to the question-only control.
The split does not satisfy any criterion requiring zero text-only content
overlap, and it must not be described as a sealed or blind audit.

The 564 uploader groups support the requested 264/100/200 allocation, but
independent audio assets are not proven until waveform/content checks. Exact
pretraining overlap with the candidate audio encoder is also unknown. Thus this
is a split-feasibility result, **not full data admission and not authorization
to train**.

## Exposure recheck: reject for new quality work

After generating the metadata-only split, the prior [Clotho-AQA result](R2_CLOTHO_AQA_RESULTS_20261013.md)
was rechecked. It already used this exact 3,422-question / 856-source eligible
pool: 2,312 questions / 578 sources for fitting, 550 / 138 for calibration,
and 560 / 140 for validation. The joint checkpoint was selected on that
validation set. Therefore this 1,252-row development partition reuses examples
and audio assets whose labels, features, or model outcomes have already
influenced development. Its clean uploader grouping relative to the newly
proposed membership file does not restore independence from prior work.

**Disposition: reject Clotho-AQA for another clean quality evaluation or tuning
run. Do not acquire more Clotho media or train on this repartition.** Keep the
new output only as a split-feasibility / exposure-audit record. The earlier
joint result remains a mixed, exposed development result, not a clean quality
gate.

## Audit implementation and failed attempt

`scripts/audit_clotho_aqa_group_split.py` pins the input SHA-256, checks the
exact clip-license allowlist, vote count/values and majority target, hashes
source identifiers, and refuses to overwrite output. Its first local execution
revealed that votes are stored as `yes`/`no` strings rather than integers; it
stopped before creating output. The parser was corrected to accept only exact
binary strings or integers, then the audit succeeded twice with identical
membership. This was an audit-script input-format error, not a data or model
failure.

Focused verification: Ruff passed; five CPU tests passed. Tests cover deterministic
group assignment, uploader/asset/source disjointness, repeated-question reporting,
vote preservation, duplicate sample rejection, license rejection, and target
consistency. The full suite was not run in this step.

## Next gate

Before any feature extraction or fitting, independently establish waveform
identity and permitted media availability for the exact clip list; audit
audio overlap with other project corpora and the chosen pretrained encoder;
freeze how three-vote disagreement enters training and scoring; and define a
matched question-only/audio-only/joint evaluation contract. If the textual
question overlap is incompatible with the project leakage gate, this task must
be rejected or explicitly scoped as in-distribution question wording rather
than silently filtered after looking at outcomes.
Because the pool is already exposed, these steps are **not** authorization to
continue with Clotho-AQA; they apply only to a genuinely new, unexposed source.

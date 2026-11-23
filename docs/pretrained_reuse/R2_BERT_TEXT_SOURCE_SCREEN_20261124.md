# R2 BERT Text Candidate Source Screen — 2026-11-24

## Decision

The Digital Agency's Japanese law multiple-choice QA release is a potentially useful *narrow legal-domain* four-option Decision task, but it is **not admitted for a BERT quality experiment** in its current form. Its official source file has only 140 rows, represents three statutes, contains repeated normalized question/context groups, and publishes no independent train/validation/test splits. A row-random split would risk context leakage; grouping by statute leaves only three independent groups. The dataset therefore cannot support a credible train/validation/sealed-audit design or a general Japanese text Decision claim.

The larger Kyoto University Commonsense Inference dataset (KUCI) is structurally a much better Japanese four-choice task, but its official repository declares CC BY-SA 4.0. The current project license policy only admits a small explicit permissive set and does not include CC BY-SA. KUCI remains `REVIEW_NOT_ADMITTED`; this screen did not download its payload or inspect examples.

No model, training, validation, or evaluation was run. The source has no official split, and its `output` answer field was present in the fetched JSON but was not enumerated, summarized, or scored. No project sealed-audit data was accessed. The root worktree's pre-existing user changes were left untouched.

## Digital Agency law-QA source facts

- Official source: `digital-go-jp/lawqa_jp`, branch `main`, commit `fce599797ce9b582045a32a5710d9197e7200959` (observed 2026-11-24).
- Source file: `data/selection.json`, Git blob `90b5e2ca40a6a5c104216f2d28a708342173258c`, 381,345 bytes.
- The source README describes a four-choice Japanese law QA dataset with context, question, options, answer key, and references. It says questions were created and checked with multiple LLMs and are not legal advice.
- Dataset metadata inspected in memory only: 140 rows; 140 unique filename IDs; 125 unique normalized question strings; 114 unique normalized contexts; 26 repeated-context groups; 2 repeated-question groups covering 17 rows.
- Grouping the filename's statute prefix yields only three underlying statute groups: 金商法 80, 薬機法 39, 借地借家法 21. Reference hosts include e-Gov, FSA, MHLW, and MLIT.
- No answer labels were enumerated or summarized, no sample content was printed into this report, and no model scores were produced. The JSON payload was fetched to memory to count structure and group/duplicate metadata.

## Rights disposition

The Digital Agency release notice says the dataset uses Public Data License 1.0 (PDL 1.0), described as compatible with CC BY 4.0 and intended to allow reuse including commercial use. The official PDL requires attribution and a notice when material is edited. It also says third-party material and linked external-source content remain subject to their own rights/terms. Since this dataset cites external legal sources, source-level rights and attribution must be checked before any training ingestion. The current repository's exact license allowlist does not include a `PDL-1.0` alias, so the repository gate would currently return REVIEW rather than ALLOW.

This report records source facts, not a legal conclusion or permission to train.

## Candidate comparison

| Candidate | Scale / format | Rights | Disposition |
|---|---|---|---|
| Digital Agency law QA | 140 four-choice questions; 3 statute groups; no official split | PDL 1.0; external references need their own review | Not admitted: insufficient independent groups and no viable sealed split |
| KUCI | Official README reports 83,127 train / 10,228 dev / 10,291 test four-choice examples | CC BY-SA 4.0 | Not admitted under current explicit allowlist; no payload downloaded |
| JamC-QA-V2 | Japanese multiple-choice knowledge benchmark | CC BY-SA 4.0 | Not admitted under current explicit allowlist; no payload downloaded |

A BERT-family run on the 140-row legal set would be an underpowered pilot and is unlikely to resolve whether BERT helps general text Decision. Do not spend model-training or evaluation budget on it. If a future rights review admits KUCI or another large, source-grouped task, compare one pinned compact BERT-family candidate against the existing text tower with an identical option scorer, a frozen group split, full resident-parameter accounting, and a predeclared validation-only selection rule.

## Primary sources

- Digital Agency release notice: https://www.digital.go.jp/en/news/9d06cea7-4b62-4d27-b036-d39cf65dad73
- Official dataset repository and schema: https://github.com/digital-go-jp/lawqa_jp
- Official PDL 1.0: https://www.digital.go.jp/resources/open_data/public_data_license_v1.0
- KUCI official repository, task statistics, and license: https://github.com/ku-nlp/KUCI
- JamC-QA-V2 dataset card and declared license: https://huggingface.co/datasets/sbintuitions/JamC-QA-V2


# AVQA source and task screen — 2026-11-15

## Disposition

**`REVIEW_NOT_ADMITTED` for product-directed Path A work.** The task format is
promising, but the official source does not establish rights for product use
of its source videos, and the both-modality category is too small for the
project's provisional independent-group gate. No annotations or media were
downloaded, and no model, features, or sealed evaluation data were accessed.

## Official source evidence

- The authors' [AVQA project page](https://mn.cs.tsinghua.edu.cn/avqa/) states
  that clips are sourced from VGG-Sound and provides YouTube IDs, timestamps,
  and train/test split labels. It describes multiple-choice questions with
  four answers and a `question_relation` field distinguishing View, Sound,
  and Both.
- The same page reports 2,900 videos and 1,661 questions. Its category table
  lists only 22 Audio-Visual examples; this is below the provisional minimum
  of 200 independent groups even before source-video grouping is checked.
- The page's license section reproduces ACM's personal-use terms: copying is
  limited to non-commercial purposes, and redistribution requires specific
  permission. It does not grant the project product-directed rights to the
  underlying YouTube videos. The page gives no immutable source revision or
  per-video rights inventory.

## Task-fit and gate result

| Gate | Finding | Disposition |
|---|---|---|
| Supplied-option Decision | Four-choice annotations and modality-relation labels are documented | Structurally promising |
| Rights for intended use | ACM page terms are non-commercial; each source video is from YouTube and per-video rights are not established | Fail closed |
| Independent groups | 22 Audio-Visual examples total; distinct source-video count and split grouping were not checked | Insufficient / unknown |
| Both-input necessity | The `Both` label is author-provided, but no matched unimodal controls were assessed here | Not established |
| Exposure | No AVQA-specific payload or project artifact was accessed during this screen | Metadata-only; not a clean-data certificate |

Do not download or fit AVQA for this product-directed track. Reconsider only
if the source owners establish applicable product-use rights for the exact
video assets and provide a sufficiently large source-grouped both-required
cohort. This screen is not quality evidence and does not authorize reopening
any other exposed audio-video validation set.

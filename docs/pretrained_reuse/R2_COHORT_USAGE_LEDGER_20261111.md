# R2 source cohort usage ledger — 2026-11-11

## Change

Added `scripts/check_cohort_usage.py` and
`manifests/pretrained_reuse/cohort_usage_ledger.json`. The checker compares
source-scoped, revision-scoped group identity hashes before a cohort is used.
It rejects known overlap, unknown source/revision, duplicate group IDs, and any
candidate from a source whose historical inventory is not marked complete.
Raw group IDs are read from an input JSON/JSONL file but only SHA-256 identity
hashes are stored in the ledger.

## Reconciled evidence

The ledger registers one verified, already-observed development cohort:

- Source: `google-deepmind/perception_test`
- Source revision: `3938d2f1ba3a6b502025741cea4cd73c7b3bdfaf`
- Task: video-plus-text multiple-choice readout
- Saved predictions: 2,886 rows from 1,518 unique video groups
- Prediction file SHA-256:
  `efa74a3596164aa343ce1ddc4e350987f178b25328c634eeb1b7f283f3304212`
- Evaluation report SHA-256:
  `d3e5b1084c3109afe83b9d3f9fee21d386bb705aaac1e49231d67b88eeecd5d2`
- Stored cohort identity hashes: source/revision/video-ID scoped SHA-256 values

The ledger marks this source's history `INCOMPLETE`. That is deliberate: the
registered evaluation is proven, but the cross-worktree inventory does not
prove that every historical training, selection, or metadata exposure is
represented. Consequently, even a disjoint-looking group is blocked until a
complete source history audit is explicitly documented. This prevents a
partial ledger from being mistaken for proof of novelty.

## Use

```powershell
python scripts/check_cohort_usage.py `
  --ledger manifests/pretrained_reuse/cohort_usage_ledger.json `
  --source-id google-deepmind/perception_test `
  --revision 3938d2f1ba3a6b502025741cea4cd73c7b3bdfaf `
  --groups proposed-groups.jsonl
```

The current command is a preflight primitive; it is not yet wired into every
training/evaluation launcher. Its result must be recorded with the run until
the launcher integration is complete. `CLEAR_WITHIN_DECLARED_HISTORY` proves
only a non-overlap within a source inventory explicitly marked complete. It
does not prove semantic novelty, label independence, rights, or sufficient
sample size.

## Verification and scope

- `python -m pytest tests/test_check_cohort_usage.py -q`: 7 passed.
- `ruff check scripts/check_cohort_usage.py tests/test_check_cohort_usage.py`:
  passed.
- Synthetic CLI cases cover known overlap and incomplete-history blocking.
- The saved prediction artifact was read only to collect group IDs. No model,
  source media, official test split, or sealed audit was opened. No GPU work,
  training, or new quality evaluation was performed.

This is R2 exposure-control infrastructure only. It does not complete R2 task
admission, the multimodal quality gate, or any later R0/R3–R8 requirement.

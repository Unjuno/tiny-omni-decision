# AIST Bimanual metadata follow-up — 2026-11-16

## Disposition

Keep AIST Bimanual `REVIEW_NOT_ADMITTED` for training or quality evaluation.
The official downloader metadata helps interpret the archive inventory, but it
does not resolve the official count discrepancy, identify independent
episode/session groups, or establish a non-leaking supplied-option Decision
task. No dataset archive or Dropbox URL was opened, probed, or downloaded.

## What the metadata establishes

- The official downloader repository documents **117 files** in its example
  index. Example archive sizes in its README are approximately 2.34, 3.04, and
  3.12 GB. These are examples, not an estimate of the whole release. The
  actual archive sizes and a suitable small subset remain unknown.
- The official task CSV is task/archive metadata, not a split manifest. Its
  columns do not provide split, episode, participant, session, or recording
  group IDs. The previously screened CSV contained 119 rows and summed to
  12,025 in its `HDF5 count` column.
- The official landing page's task table had 117 rows and reported 10,705
  total episodes, while the official details page says “117 episodes.” These
  counts do not agree with one another or the CSV sum. The downloader's 117
  files are consistent with archive/task-level files, but do not establish
  that the underlying episode counts are reconciled.
- The task metadata includes action descriptions/prompts. Until a fixed task
  and controls show that the prompt does not reveal the target, prompt leakage
  remains a material risk.

## Gate decision

| Gate | Current evidence | Result |
|---|---|---|
| Rights | Official landing page states CC BY 4.0 | Promising; pin attribution and source version before any future use |
| Synchronized modalities | RGB/depth, robot state/action, and task text are described | Structurally promising; no payload checked |
| Independent validation | No session/episode split identities in the public task CSV | Fail closed; the provisional 200 independent validation group floor is unverified |
| Target integrity | Action/task prompt may reveal the action target; no frozen option-choice protocol or modality controls | Not established |
| Local feasibility | README examples are multi-GB; total sizes and subset behavior are unknown | Do not acquire archives for screening |
| Inventory | Landing page, details page, and CSV report incompatible counts | Unresolved |

Do not create random train/validation partitions from task counts or dates.
Reopen this candidate only if the provider supplies a reconciled inventory, a
defensible group/split manifest meeting the validation floor, and enough task
information to predefine a leakage-resistant target with modality controls.

## Source records and boundaries

- [Official AIST landing page](https://aistairc.github.io/aist_bimanip_site/index.html)
- [Official dataset details](https://aistairc.github.io/aist_bimanip_site/dataset.html)
- [Official downloader repository](https://github.com/aistairc/aist_bimanip_downloader)
- The CSV and rendered-page hashes, retrieval date, and initial gate analysis
  are recorded in the [source screen](R2_AIST_BIMANUAL_SOURCE_SCREEN_20261116.md)
  on the read-only screening branch at source commit
  `bfb13bf6e9ac4f2ce244eb51e470afa907587380`.
- This follow-up inspected public downloader documentation only. No Dropbox
  link was followed or queried; no archive, media, trajectory, annotation
  payload, model, or evaluation label was accessed. No training or GPU work
  was run.

## Development consequence

AIST Bimanual does not currently supply an admitted R2 task, so it does not
justify a new fit or quality claim. Continue engineering against the existing
Path A runtime and reproducibility gates, and keep the real multimodal M-2
quality gate open until a task passes the source, split, and leakage checks.

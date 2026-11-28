# Physion++ official train-label mapping audit — 2026-11-28

## Scope and purpose

This follow-up checks whether every example in the existing `fullclip-v1`
**train split** points to the matching official Physion++ readout metadata and
whether the corpus label matches both the metadata label and its frame-level
contact timeline. It is a data-integrity check, not a model-quality evaluation
or an independent human relabeling.

Only `C:\\CodexArtifacts\\tqpp\\fullclip-v1\\train.jsonl` was read. Its SHA-256
was pinned to
`0de1bbbb1cc54000936c1771ea54e375c214fedda36734536f31a44c4f64932b`.
Validation and sealed/test rows were not loaded or evaluated. No model or GPU
was used.

The publisher's data description says the readout `.pkl` records include the
video label, object identities, and physical event metadata; frame JSON holds
object IDs and masks over time. See the [official NeurIPS supplement](https://proceedings.neurips.cc/paper_files/paper/2023/file/d3e8011c912e651ab2a76e7935a1e464-Supplemental-Datasets_and_Benchmarks.pdf)
and [project page](https://dingmyu.github.io/physion_v2/).

## Pinned source and procedure

The audit used the official
`https://physion-v2.s3.amazonaws.com/readout_data.zip` archive, pinned by
ETag `"4cf84f667bcc3eaf991dbcc10dd31177-355"` and size 2,971,891,321 bytes.
The central directory contained 5,073 entries. For each of 652 train rows,
the code derived the exact `.pkl` member from the full media path, then checked
conditional HTTP Range responses, ETag, `Content-Range`, local/central ZIP
headers, decompressed byte count, and CRC-32.

Pickle decoding is restricted to NumPy array/dtype/scalar reconstruction and
the built-in `bytes` and `str` types. Other pickle globals fail closed. No
pickle-supplied arbitrary functions or classes were allowed.

For each row the audit compared:

1. corpus `target` and `source_target`;
2. official `static.does_target_contact_zone`;
3. whether any official frame has
   `labels.target_contacting_zone == true`;
4. official `static.stimulus_name` against the expected scenario and trial
   identity derived from the exact ZIP member path.

## Results

| Check | Result |
|---|---:|
| Train examples checked | 652 / 652 |
| Exact official metadata members | 652 / 652 |
| Unique official metadata SHA-256 values | 652 |
| Corpus target vs official trial label mismatches | 0 |
| Corpus source_target vs official trial label mismatches | 0 |
| Official trial label vs frame contact timeline mismatches | 0 |
| ZIP-member identity vs official stimulus-name mismatches | 0 |
| Validation or sealed/test rows opened | No |

The manifest contains 339 No / 313 Yes; official metadata yields the same
counts. Counts by scenario are:

| Scenario | No | Yes | Total |
|---|---:|---:|---:|
| `bouncy_platform_pp` | 55 | 32 | 87 |
| `bouncy_wall_pp` | 43 | 46 | 89 |
| `deform_clothhang_pp` | 45 | 39 | 84 |
| `deform_clothhit_pp` | 32 | 52 | 84 |
| `friction_cloth_pp` | 28 | 28 | 56 |
| `friction_collision_pp` | 28 | 22 | 50 |
| `friction_platform_pp` | 34 | 17 | 51 |
| `mass_collision_pp` | 21 | 32 | 53 |
| `mass_dominoes_pp` | 29 | 17 | 46 |
| `mass_waterpush_pp` | 24 | 28 | 52 |
| **Total** | **339** | **313** | **652** |

## Interpretation and limits

This closes the specific train-corpus serialization/mapping question: each
video row resolves to the official metadata for that same stimulus, and the
manifest labels reproduce the publisher's trial label and contact timeline.
It materially lowers the likelihood that the observed train labels were
shifted to a different stimulus or inverted during corpus construction.

The check is not independent of the simulator metadata that produced the
labels. It does **not** establish that every label is visually obvious from
the RGB video, that an independent annotator would agree, or that the model
cannot exploit scenario, rendering, or simulator-factor shortcuts. The
existing train-only factor skew and the need for blinded visual review remain.
No validation or sealed-audit claim follows from these results.

## Reproduction artifacts

- Audit code: `scripts/audit_physionpp_official_train_labels.py`
- Unit tests: `tests/test_audit_physionpp_official_train_labels.py`
- Source commit: `09195fe96d1b4ff2bc06e7678be59dffd28b49e4`
- Audit script SHA-256 (as recorded by the successful run):
  `2b8fa399ff40c15e6e5132139ed19d4ca22b37e2a5b1f7dfbcebcfc9e285d8f9`
- Frozen train manifest SHA-256:
  `0de1bbbb1cc54000936c1771ea54e375c214fedda36734536f31a44c4f64932b`
- Full per-row results:
  `C:\\CodexArtifacts\\pretrained-reuse-r2-local\\physionpp-label-integrity-20261128-attempt-06\\train-label-verification.jsonl`
  (SHA-256 `354175780c48929c1be1c137bd76b4af0520f10605a3ed1b75237c350068dfdb`)
- Machine report:
  `C:\\CodexArtifacts\\pretrained-reuse-r2-local\\physionpp-label-integrity-20261128-attempt-06\\audit-report.json`
- Per-scenario counts:
  The successful attempt-06 machine report records the verified per-scenario
  counts above; no separate `scenario-label-summary.json` was emitted in that
  attempt.

Attempts 01–05 and their partial or superseded outputs are preserved. Attempt 01's positive
`0003` sample was invalid for its intended pair because a basename-only lookup
selected a different scenario; its `run-status.json` marks those files as
unsuitable evidence. Attempts 03 and 04 each stopped fail-closed on a newly
encountered pickle/type assumption; their verified partial files were
preserved. Attempt 05 completed the verification, and attempt 06 reran all
652 records under the final event-index-preserving schema. The report and
output hashes above refer to attempt 06 only.

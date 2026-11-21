# PR #94 Path A runtime: independent CPU review — 2026-11-15

## Scope

Reviewed PR #94 head `6d722c258f188a0a09c624bf7b39e3f352d55319` in a clean,
detached worktree at `C:\CodexWorktrees\path-a-pr94-review-20261115`. The
primary checkout and PR branch were not edited. No models, media, checkpoint,
or GPU inference were run.

The review covered `path_a_jsonl.py`, the typed adapter in
`runtime_dispatch.py`, the JSONL protocol tests, and the temporal-session tests.
The CLI, request parsing, typed option binding, finite-output serialization,
and bounded in-memory feature-session behavior were inspected. This is a
focused contract review, not a complete security review or a new quality result.

## Verification

- `python -m pytest tests/test_path_a_jsonl.py tests/test_path_a_temporal_session_jsonl.py -q`
  — **16 passed** in 0.68 s.
- `ruff check src/tiny_omni_decision/path_a_jsonl.py tests/test_path_a_jsonl.py tests/test_path_a_temporal_session_jsonl.py`
  — **passed**.
- Python 3.11.9 and pytest 8.4.2 from the existing environment were used; no
  packages were installed or upgraded.
- PR #94's own current CI had already passed in the live snapshot, but that
  does not substitute for task-quality evidence.

No contract defect was found in the inspected paths. Tests verify malformed
request rejection, option-order binding, one audio decision per query, typed
JSON output, line-numbered JSON errors, process-local session limits, causal
snapshot behavior, explicit missingness, stale-input handling, and bounded
history eviction.

## Product interpretation

This is useful Path A runtime plumbing: it binds caller-supplied option IDs to
existing task-worker probabilities and exposes typed JSON. The session API
stores caller-supplied hash-bound features in a process-local buffer; it is not
an encoder, learned fusion model, persistent world representation, or evidence
that temporal state improves decisions. The measured four-route replay in the
PR report is not an Accuracy/NLL/Brier/ECE evaluation. That report also records
the current assembled FP32 package at about 687 MB, above the provisional
200 MB quantized-package target; no quantized artifact was tested.

Keep PR #94 as a plumbing candidate. Product-quality and actual quantized
package gates remain open; do not infer them from this review.

## Separate evidence disposition

Clotho-AQA is not a fresh next task. The tracked 2026-10-13 result already
compares question-only, audio-only, and joint frozen-feature readouts on a
grouped development split. Joint Accuracy exceeded question-only by 2.14
points, but the paired uploader interval included zero and calibration was
slightly worse; later epochs overfit. No official AQA validation/test labels or
sealed audit were opened. Preserve those results and do not retune on them.
See `R2_CLOTHO_AQA_RESULTS_20261013.md` and
`R2_CLOTHO_AQA_ADMISSION_20261013.md`.


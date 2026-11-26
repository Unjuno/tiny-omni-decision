# R2 Path A typed-video option-order guard — 2026-11-04

## Finding

The Path A video scorer has a fixed binary class order: `wrong`, then
`correct`. `dispatch_typed("video", ...)` previously accepted caller option
texts in another order, even though the scorer's returned probability vector
and predicted index remained in the fixed worker order. This could bind a
probability to the wrong caller option ID in the typed JSON result.

## Change

The typed dispatcher now rejects video specs unless the option text order is
exactly `("wrong", "correct")`, before either worker starts. It does not
change the scorer, weights, dataset, prediction, confidence, or task quality.
The matching-order regression verifies that the `correct` probability maps to
the caller's correct option ID; the reversed-order regression verifies the
request is rejected before any worker starts.

## Verification and limits

- `python -m pytest tests/test_runtime_dispatch.py -q`: **8 passed**.
- Ruff on the changed source and test files: passed.
- `git diff --check`: passed.
- CPU-only contract validation; no model loaded, GPU used, or validation data
  evaluated. This fixes typed-output integrity only and adds no quality claim.

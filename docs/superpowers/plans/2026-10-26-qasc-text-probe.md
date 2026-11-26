# QASC Frozen-Feature Text Decision Probe Implementation Plan

> **For agentic workers:** Execute this plan task-by-task. Each code change is test-first and stays on the PR #15 worktree.

**Goal:** Add a provenance-preserving QASC text Decision path and measure the existing frozen MiniLM + shared option scorer on a new, group-checked task without opening the official test shard.

**Architecture:** A small QASC adapter converts official rows into the repository's generic text Decision schema. The existing MiniLM feature extractor and `FrozenFeatureCandidateScorer` remain unchanged; the runner gains an opt-in generic adapter path while its current synthetic default remains byte-for-byte behaviorally compatible. Artifacts and downloaded data remain outside Git.

**Tech Stack:** Python 3.11, PyArrow from the optional data environment, existing Pydantic dataset schema, PyTorch/Transformers, pytest.

**Spec:** `ROADMAP.md` R0/R2 and the user-provided Pretrained World Representation Reuse objective.

## Global Constraints

- Source revisions: AI2 QASC dataset mirror `a34ba204eb9a33b919c10cc08f4f1c8dae5ec070`; official source repository `allenai/qasc@c44a320c2bb682d614a675d2ba4be072150db4eb`.
- Use only the 8,134 train and 926 validation rows; never fetch or parse the test shard.
- Record CC-BY-4.0 attribution and the exact downloaded shard SHA-256 values.
- Do not pass `fact1`, `fact2`, or `combinedfact` into the model; use the supporting-fact pair only as a group identity.
- Do not alter existing model checkpoints, corpora, artifacts, other worktrees, or active GPU processes.
- Keep the experiment on local compute with cloud cost $0; use the cooperative HotGPU guard if a GPU run is justified.
- Preserve the existing `train_frozen_text_probe` default behavior and synthetic-task claims.
- Treat QASC validation as development/model-selection data, not a blind audit.

## Review Focus

1. Malformed or reordered choice labels must not silently point to the wrong target (adapter fixture tests).
2. Rows sharing a supporting-fact pair must share one validation group (group-identity tests and split audit).
3. Facts used only for grouping must not enter `DecisionExample.state` or `question` (adapter privacy/input-boundary tests).
4. A missing or mislabeled train/validation manifest must fail closed (runner manifest checks).
5. The test shard must remain an unresolved/unopened artifact, recorded by LFS pointer only (preparation receipt).

### Task 1: QASC row adapter

**Files:**
- Create: `src/tiny_omni_decision/qasc.py`
- Test: `tests/test_qasc.py`

**Interfaces:** `normalize_qasc_row(row: dict[str, Any], *, revision: str, split: str, license_name: str = "CC-BY-4.0") -> dict[str, Any]` returns the repository generic-row shape with original choice order, text-only input, and a fact-pair group prefix on `source_record_id`.

- [ ] Write tests for valid label-to-choice mapping, fact-order-invariant group IDs, omission of facts from model inputs, and malformed/missing choices failing closed.
- [ ] Run the new tests and confirm expected failures.
- [ ] Implement the smallest adapter satisfying those tests.
- [ ] Run the tests and Ruff on the touched files.

### Task 2: Shard preparation and source manifests

**Files:**
- Create: `scripts/prepare_qasc_path_a.py`
- Create: `manifests/candidates/qasc-train.yaml`
- Create: `manifests/candidates/qasc-validation.yaml`
- Test: extend `tests/test_qasc.py` for split/group/content collision checks.

**Interfaces:** The preparation CLI consumes pinned train/validation Parquet paths and expected SHA-256 values, writes JSONL and a receipt only to a fresh external output directory, and refuses to read any test path.

- [ ] Add tests for the split audit failing when a question or fact-pair group crosses splits.
- [ ] Run tests to verify failures.
- [ ] Implement preparation, ordering hashes, per-split counts, and training/evaluation manifests.
- [ ] Run tests and prepare only train/validation rows from the pinned local mirror.

### Task 3: Generic frozen text-probe runner path

**Files:**
- Modify: `scripts/train_frozen_text_probe.py`
- Test: `tests/test_qasc.py` or a focused runner test.

- [ ] Add a failing test for the generic adapter path and separate training/evaluation manifests.
- [ ] Implement optional generic input mode, task claim metadata, and fail-closed manifest/source checks while preserving defaults.
- [ ] Run targeted and full CPU tests plus Ruff; keep ML execution outside normal CI.

### Task 4: Preflight, train, analyze, document

**Files:**
- Create: `docs/pretrained_reuse/R2_QASC_TEXT_DECISION_20261026.md`
- Modify: `ROADMAP.md` with a minimal evidence entry.

- [ ] Record source/processor revisions, effective config, hashes, group counts, leakage checks, and metadata-only controls before fitting.
- [ ] Run a distinct-process smoke and only then one seed-17 frozen-feature readout fit, with the selected checkpoint determined by the frozen validation-NLL rule.
- [ ] Save all predictions, metrics, learning history, hashes, parameter counts, timing, and resource measurements externally.
- [ ] Report task- and group-aware validation results as development evidence; never access the sealed/official test shard.
- [ ] Re-run Ruff, pytest, manifest/catalog validation, check the diff, commit, push, and inspect CI.

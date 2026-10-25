# Independent AV Readout Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compare a temporal LSTM and a query-conditioned fusion head on a new controlled AV sequence-matching task without reusing exposed evaluation data.

**Architecture:** Generate matched and permuted audio/video symbol sequences with balanced query polarity, render them into eight-frame videos and eight-tone audio clips, then extract frozen feature sequences with pinned VideoPrism and EfficientAT. Fit two small matched readouts on the same feature cache and evaluate fixed input-necessity/time-order controls.

**Tech Stack:** Python 3.11, NumPy, PyTorch, PyAV, pinned VideoPrism JAX/TensorFlow runtime, pinned EfficientAT PyTorch runtime, pytest, Ruff.

**Spec:** `docs/superpowers/specs/2026-10-25-independent-av-readout-design.md`

## Global Constraints

- Use seed 17 and generated data only; never load exposed CLEVRER, Perception Test, HoloAssist, or any sealed audit payload for this comparison.
- Keep VideoPrism and EfficientAT frozen; do not change their revisions, precision, or preprocessing after feature extraction begins.
- Use a new, empty external artifact directory for each stage; never overwrite prior outputs.
- Use the HotUGPU reservation wrapper for all CUDA work; never stop or modify Orbitune processes.
- Keep train, validation, and test data grouped by complete sequence-pair group; every sequence identity stays within one split.
- Compare the two heads with the same feature cache, sample order, seed, optimizer, total update budget, validation schedule, and best-checkpoint selector.
- Select checkpoints by validation NLL only; keep terminal checkpoints separate.
- Report this as a synthetic architecture diagnostic only; do not claim natural AV or product quality.

## Review Focus

- Query polarity can leak the answer if both polarities are not emitted for every pair; test exact complementary-label pairing.
- Audio or video sequence identities can cross splits through repeated pair construction; test group, sequence, normalized content, and asset disjointness.
- A task can be solvable from symbol frequency without temporal order; test same-symbol-multiset permutations and exact order targets.
- Renderer timestamp sampling can alias or drop an event; test that the pinned eight-frame decoder receives the intended eight ordered symbols.
- Missing/shuffled inputs can accidentally be replaced with valid zeros or misreported; test explicit control transforms and retain all predictions.

---

### Task 1: Deterministic balanced AV task manifest

**Files:**
- Create: `src/tiny_omni_decision/synthetic_av_task.py`
- Test: `tests/test_synthetic_av_task.py`

**Interfaces:**
- Produces `build_manifest(train_groups: int, validation_groups: int, seed: int = 17) -> dict[str, object]`.
- Produces `validate_manifest(manifest: dict[str, object]) -> None`.
- Each group contains two distinct length-eight symbol sequences with identical symbol multisets; emit AA, BB, AB, BA input pairs and both `same_order` / `different_order` queries with exact deterministic binary targets.

- [ ] Write tests for deterministic ordering, complementary query targets, balanced audio-only/video-only marginals, same-multiset hard negatives, and complete group/sequence split isolation.
- [ ] Run `python -m pytest tests/test_synthetic_av_task.py -q`; expect failure because the module does not exist.
- [ ] Implement the smallest manifest builder and fail-closed validator.
- [ ] Re-run the targeted tests; expect all pass.
- [ ] Run `python -m ruff check src/tiny_omni_decision/synthetic_av_task.py tests/test_synthetic_av_task.py`.
- [ ] Commit as `feat: add balanced synthetic AV sequence task`.

### Task 2: Reproducible audio/video renderer and CPU smoke

**Files:**
- Create: `scripts/prepare_synthetic_av_readout.py`
- Modify: `src/tiny_omni_decision/synthetic_av_task.py`
- Test: `tests/test_synthetic_av_task.py`

**Interfaces:**
- Produces `render_assets(manifest: dict[str, object], output_dir: Path) -> dict[str, object]` and refuses an existing output directory.
- Writes one manifest, eight ordered frame images per unique video sequence, and one WAV per unique audio sequence. Persist per-file SHA-256, dimensions, sample rate, generated tone-bin ID, and parent group/split identity.

- [ ] Add tests that rendering is deterministic for a fixed seed, different sequences produce different asset hashes, sample rates and durations are exact, and output creation refuses overwrite.
- [ ] Run the targeted tests and confirm the new tests fail before the renderer exists.
- [ ] Implement NumPy/Pillow frame rendering and PCM tone synthesis using only already-installed packages; do not install or upgrade dependencies.
- [ ] Run an 8-group CPU smoke into a fresh `C:/CodexArtifacts/pretrained-reuse-r2/synthetic-av-readout-smoke-01` directory; verify media hashes, frame order, dimensions, WAV format, and split checks.
- [ ] Run Ruff and targeted tests; commit as `feat: render synthetic AV sequence assets`.

### Task 3: Frozen feature extraction preflight and full cache

**Files:**
- Create: `scripts/extract_synthetic_av_readout_features.py`
- Create: `configs/pretrained_reuse/synthetic_av_temporal_vs_conditional_v1.json`
- Test: `tests/test_synthetic_av_task.py`

**Interfaces:**
- Consumes renderer manifest and exact VideoPrism/EfficientAT checkpoint paths and hashes.
- Produces finite, time-indexed feature arrays keyed by immutable sequence ID, plus an effective config, environment record, model/source revision, parameter counts, and artifact hashes.
- Implements separate smoke and full-cache modes; CUDA execution requires the HotUGPU reservation variable and fails closed otherwise.

- [ ] Test hash/schema validation, ordered frame-count checks, finite outputs, and CUDA guard behavior without initializing CUDA.
- [ ] Run tests and confirm new assertions fail before the extractor implementation.
- [ ] Implement model loaders by reusing the pinned project inference helpers; keep both encoders in eval/frozen mode and preserve frame/audio timestep alignment.
- [ ] Run one-batch smoke under the CPU path first; verify output shapes, dtypes, finite values, model state hashes before/after, and encoder parameter counts.
- [ ] Run full feature extraction only through `scripts/run_with_hotgpu_guard.py` into a new attempt directory; if the guard waits, allow it to wait without stopping other processes.
- [ ] Verify the feature cache and source/media/config hashes in a fresh CPU process; commit as `feat: extract frozen features for synthetic AV task`.

### Task 4: Matched temporal and conditional readouts

**Files:**
- Create: `src/tiny_omni_decision/synthetic_av_readouts.py`
- Create: `scripts/train_synthetic_av_readouts.py`
- Modify: `configs/pretrained_reuse/synthetic_av_temporal_vs_conditional_v1.json`
- Test: `tests/test_synthetic_av_readouts.py`

**Interfaces:**
- `TemporalLSTMReadout(audio_width: int, video_width: int, query_width: int, hidden_size: int)` consumes ordered aligned features and query polarity and returns two logits.
- The temporal head concatenates audio, video, and the repeated query embedding at each of eight time steps before its LSTM.
- `ConditionalAVReadout(audio_width: int, video_width: int, query_width: int, hidden_size: int)` applies query-to-sequence attention separately to audio and video, then a learned two-way gate and shared binary scorer.
- Match trainable counts within 10%; report exact count and clarify that gate fusion is conditional weighting, not iterative cross-modal message passing.
- Trainer uses one seed, common ordered examples, AdamW (LR 0.001, weight decay 0.0001), batch 32, 128 epochs (16,384 optimizer updates for 4,096 training rows), validation every 4 epochs, and validation NLL best selection; terminal weights stay separate.

- [ ] Test output shapes, nonfinite rejection, gradient reachability, exact trainable-parameter counts, and deterministic initialization.
- [ ] Run tests and confirm they fail before model classes exist.
- [ ] Implement both heads with readout parameter counts within 10%; record exact counts and input tensor requirements.
- [ ] Implement fixed-budget training and checkpoint serialization; test best/terminal separation and that neither architecture receives different sample order or update counts.
- [ ] Run a separate-process 2-group training smoke on CPU; verify reload reproduces logits exactly and encoders are not present in optimizer parameters.
- [ ] Run Ruff and targeted/full CPU tests; commit as `feat: compare temporal and conditional AV readouts`.

### Task 5: Frozen validation controls, uncertainty, and result publication

**Files:**
- Create: `scripts/evaluate_synthetic_av_readouts.py`
- Create: `docs/pretrained_reuse/R2_SYNTHETIC_AV_READOUT_COMPARISON_20261025.md`
- Modify: `ROADMAP.md`
- Test: `tests/test_synthetic_av_readouts.py`

**Interfaces:**
- Evaluator scores both fixed best checkpoints and terminal checkpoints on identical ordered validation IDs, saves per-example probabilities, and runs audio-only/video-only/missing/shuffled/reversed controls without selecting from controls.
- Uncertainty uses paired bootstrap resampling over complete sequence-pair groups, seed 17, 10,000 replicates, percentile 95% intervals.

- [ ] Test metric aggregation, complement-pair accuracy, intervention provenance, and deterministic group bootstrap.
- [ ] Run tests and confirm the evaluator tests fail before implementation.
- [ ] Implement the evaluator and separate report builder; include runtime, GPU/VRAM, wall time, artifact hashes, failures, and limitations.
- [ ] Run evaluation once on the fixed validation cohort and independently replay saved predictions/metrics on CPU.
- [ ] Run Ruff, full CPU tests, CI-equivalent manifest/catalog validation, and `git diff --check`.
- [ ] Update ROADMAP with the synthetic-only scope and next natural-data gate; commit as `docs: record AV readout comparison`.

## Final review

- Review the complete branch diff against the design and Review Focus.
- Ensure no media, feature cache, or checkpoint is tracked by Git.
- Push only the fast-forwardable target branches after local review, then check GitHub Actions for the exact pushed commit.

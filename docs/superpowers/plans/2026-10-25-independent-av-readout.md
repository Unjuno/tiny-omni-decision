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

- [x] Write tests for deterministic ordering, complementary query targets, balanced audio-only/video-only marginals, same-multiset hard negatives, and complete group/sequence split isolation.
- [x] Run `python -m pytest tests/test_synthetic_av_task.py -q`; expect failure because the module does not exist.
- [x] Implement the smallest manifest builder and fail-closed validator.
- [x] Re-run the targeted tests; expect all pass.
- [x] Run `python -m ruff check src/tiny_omni_decision/synthetic_av_task.py tests/test_synthetic_av_task.py`.
- [x] Commit as `feat: add balanced synthetic AV sequence task`.

### Task 2: Reproducible audio/video renderer and CPU smoke

**Files:**
- Create: `scripts/prepare_synthetic_av_readout.py`
- Modify: `src/tiny_omni_decision/synthetic_av_task.py`
- Test: `tests/test_synthetic_av_task.py`

**Interfaces:**
- Produces `render_assets(manifest: dict[str, object], output_dir: Path) -> dict[str, object]` and refuses an existing output directory.
- Writes one manifest, eight ordered frame images per unique video sequence, and one WAV per unique audio sequence. Persist per-file SHA-256, dimensions, sample rate, generated tone-bin ID, and parent group/split identity.

- [x] Add tests that rendering is deterministic for a fixed seed, different sequences produce different asset hashes, sample rates and durations are exact, and output creation refuses overwrite.
- [x] Run the targeted tests and confirm the new tests fail before the renderer exists.
- [x] Implement NumPy/Pillow frame rendering and PCM tone synthesis using only already-installed packages; do not install or upgrade dependencies.
- [x] Run an 8-group CPU smoke into a fresh `C:/CodexArtifacts/pretrained-reuse-r2/synthetic-av-readout-smoke-01` directory; verify media hashes, frame order, dimensions, WAV format, and split checks.
- [x] Run Ruff and targeted tests; commit as `feat: render synthetic AV sequence assets`.

### Task 3: Frozen feature extraction preflight and full cache

**Files:**
- Create: `scripts/extract_synthetic_av_readout_features.py`
- Create: `configs/pretrained_reuse/synthetic_av_temporal_vs_conditional_v1.json`
- Test: `tests/test_synthetic_av_task.py`

**Interfaces:**
- `scripts/extract_synthetic_av_readout_features.py video` consumes the rendered manifest plus pinned VideoPrism checkpoint/source and writes sequence-keyed `[N,8,768]` features.
- `scripts/extract_synthetic_av_readout_features.py audio` consumes rendered WAVs plus pinned EfficientAT checkpoint/source and writes four audio-symbol keyed `[4,384]` features, after verifying same-tone segment identity.
- `scripts/extract_synthetic_av_readout_features.py merge` validates both caches and writes one common cache; all stages include hashes and effective runtime metadata.
- VideoPrism uses the qualified native JAX path on Windows CPU: its saved environment reports only `TFRT_CPU_0`, and the repository's Torch port failed native-token parity on real videos. EfficientAT may use CUDA only with the HotUGPU reservation variable and fails closed otherwise.
- `--limit-groups-per-split` provides an isolated smoke subset; omitting it processes the complete fixed manifest.

- [x] Test media and cache hash/schema validation, ordered frame-count checks, finite output shape `[N,8,768]` / `[4,384]`, and CUDA guard behavior without initializing CUDA.
- [x] Run tests and confirm new assertions fail before the extractor implementation.
- [x] Implement native VideoPrism framewise extraction using the pinned JAX helper; implement EfficientAT segment extraction using its pinned evaluation mel frontend and verify all segments for one tone symbol are byte-identical; keep both encoders frozen.
- [x] Run separate-process CPU feature smokes; verify output shapes, dtypes, finite values, model state hashes before/after, and encoder parameter counts.
- [x] Complete full fixed-corpus feature extraction (VideoPrism uses the qualified Windows/JAX CPU backend; EfficientAT cache is complete).
- [x] Verify full feature cache and source/media/config hashes in a fresh CPU process.
- [x] Commit extractor/config/tests as `feat: extract frozen features for synthetic AV task`.

### Task 4: Matched temporal and conditional readouts

**Files:**
- Create: `src/tiny_omni_decision/synthetic_av_readouts.py`
- Create: `scripts/train_synthetic_av_readouts.py`
- Modify: `configs/pretrained_reuse/synthetic_av_temporal_vs_conditional_v1.json`
- Test: `tests/test_synthetic_av_readouts.py`

**Interfaces:**
- `TemporalLSTMReadout(audio_width: int, video_width: int, query_width: int, hidden_size: int = 96)` concatenates both ordered streams and repeated query input, then applies a two-layer LSTM and binary scorer.
- `ConditionalAVReadout(audio_width: int, video_width: int, query_width: int, hidden_size: int = 128, num_heads: int = 4, cross_depth: int = 2)` applies query-conditioned bidirectional audio↔video cross-attention at each time step, separate query pooling, a learned two-way gate, and a binary scorer.
- Match trainable counts within 10%; report exact count and runtime cost.
- Trainer uses one seed, common precomputed epoch permutations, AdamW (LR 0.001, weight decay 0.0001), batch 32, 128 epochs (16,384 optimizer updates for 4,096 training rows per head), validation every 4 epochs, and validation NLL best selection; terminal weights stay separate.

- [x] Test output shapes, gradient reachability, parameter-count match, and deterministic initialization.
- [x] Run tests and confirm they fail before model classes exist.
- [x] Implement both heads with readout parameter counts within 10%; record exact counts and input tensor requirements.
- [x] Implement fixed-budget training and checkpoint serialization; verify best/terminal separation and common sample order/update counts.
- [x] Run separate-process CPU training/evaluation smokes; verify best reload reproduces logits and controls serialize. Encoders are precomputed/frozen and absent from optimizer parameters.
- [x] Run Ruff and targeted CPU tests; commit training/evaluation code in separate commits.

### Task 5: Frozen validation controls, uncertainty, and result publication

**Files:**
- Create: `scripts/evaluate_synthetic_av_readouts.py`
- Create: `docs/pretrained_reuse/R2_SYNTHETIC_AV_READOUT_COMPARISON_20261025.md`
- Modify: `ROADMAP.md`
- Test: `tests/test_synthetic_av_readouts.py`

**Interfaces:**
- Evaluator scores both fixed best checkpoints and terminal checkpoints on identical ordered validation IDs, saves per-example probabilities, and runs audio-only/video-only/missing/shuffled/reversed controls without selecting from controls.
- Uncertainty uses paired bootstrap resampling over complete sequence-pair groups, seed 17, 10,000 replicates, percentile 95% intervals.

- [x] Test metric aggregation, complement-pair accuracy, intervention provenance, and deterministic group bootstrap.
- [x] Run tests and confirm the evaluator tests fail before implementation.
- [x] Implement evaluator to save per-example predictions for each validation control.
- [x] Add the full result report with runtime, GPU/VRAM, wall time, artifact hashes, failures, and limitations.
- [x] Run full evaluation once on the fixed validation cohort and independently replay saved predictions/metrics on CPU.
- [x] Run Ruff, full CPU tests, CI-equivalent manifest/catalog validation, and `git diff --check`.
- [x] Update ROADMAP with the synthetic-only scope and next natural-data gate; commit as `docs: record AV readout comparison`.

## Final review

- Review the complete branch diff against the design and Review Focus.
- Ensure no media, feature cache, or checkpoint is tracked by Git.
- Push only the fast-forwardable target branches after local review, then check GitHub Actions for the exact pushed commit.

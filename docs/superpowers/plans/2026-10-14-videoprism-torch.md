# Frozen VideoPrism GPU Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement task-by-task. Steps use checkbox syntax.

**Goal:** Execute the same licensed frozen native16-frame VideoPrism on local CUDA with validated numerical parity.

**Architecture:** Translate pinned factorized Flax operators and all40 stacked parameter arrays to a frozen Torch module. Keep patch/time order, scale+1 LayerNorm, exact GELU, query scaling and50*tanh(logit/50). Keep preprocessing external and identical for reference/port comparisons.

**Tech Stack:** Existing Torch2.6/NumPy; approved system Python plus existing JAX0.6.2/TF2.18.1 packages for references. No dependency upgrade.

**Spec:** docs/pretrained_reuse/R2_LOMBARD_GRID_VISUAL_SMOKE_20261014.md (bounded GPU optimization requirement).

## Global Constraints

- Native16x288x288 RGB[0,1],FP32,TF32 off; no interpolation/precision/architecture changes.
- Pin source c6eba0ece3dd49a24ffc44582966997a4a59f1d8 and checkpoint61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29.
- All114,365,184 learned parameters counted and frozen; no quantization/training.
- Old artifacts/layouts/runtimes untouched; new outputs. No cloud compute or live-job intervention.
- Quality scores remain unmeasured; numerical mismatch blocks full extraction.
- User's explicit autonomous execution/strategy authorization supplies inline execution approval; no redundant plan-approval stop.

## Review Focus

- Unexpected/missing/nonfinite/wrong-dtype checkpoint tensors: reject, no partial load.
- Wrong pixel range/dtype/shape or unsupported frames: reject instead of silent conversion.
- Training mode or reduced precision: fail closed; no shadow parameter updates.
- Temporal/spatial reshape order and output projection orientation: exact source convention.
- Repeated CUDA outputs and native-reference differences: report actual maxima; no threshold adjustment after seeing results.

### Task 1: Frozen operator and checkpoint implementation

**Files:** create src/tiny_omni_decision/videoprism_torch.py; tests/test_videoprism_torch.py.
**Interfaces:** Config; expected_shapes(config); patchify(video,patch_size); source_layer_norm(x,scale,bias); capped_attention(q,k,v); FrozenVideoPrism(weights,config=Config()); load_checkpoint(path).

- [ ] Write tiny independent operator/schema/order/mode tests; run pytest and observe missing-module failure.
- [ ] Implement minimal FP32 native architecture and strict source checkpoint loading.
- [ ] Run tests and Ruff; expected PASS. Commit core.

### Task 2: Native reference and real CPU/CUDA parity

**Files:** prototype audit runners under C:/CodexArtifacts/pretrained-reuse-r2; new docs/pretrained_reuse/R1_VIDEOPRISM_TORCH_20261014.{md,json}.
**Consumes:** Task1 load_checkpoint and FrozenVideoPrism.forward(B,T,H,W,C)->(B,T*N,D).

- [ ] Save identical preprocessed inputs and full native outputs for the same four training clips, plus seeded tiny operator references from pinned source.
- [ ] Compare CPU/Torch and CUDA/Torch full tokens to native results at predefined atol=1e-4,rtol=1e-4; also require mean/framewise max absolute delta<=1e-4. Require repeat/reload and exact parameter/schema accounting.
- [ ] Measure warm time/peak VRAM; no full job unless gates pass. Record failures intact and investigate before changing any gate.
- [ ] Run full CPU tests,Ruff,manifest validation; document real measurements/limits; commit/push and check target CI.
- [ ] Fresh whole-change review under requesting-code-review; fix important verified findings with tests.

## Execution ledger

Pre-flight: Task1's model/loader produces exactly the forward interface Task2 consumes. No shared-interface conflict.
Ruling: autonomous user authorization permits inline execution without another confirmation; scope preserves the native model and data.

# Readout learning mechanics and measured local cost

The full video cache continues unchanged in its original checkout. The audio
producer remains INCOMPLETE after its reference-count error; only its independently
pinned full revalidation may admit the saved audio banks. Implement an explicit
read-only loader: verify the proof hash, exact producer linkage, completed input
and feature checks, source root and every saved artifact hash. Return a cache-valid
in-memory view carrying original INCOMPLETE provenance; never edit the original
receipt. The ordinary raw incomplete-receipt rejection remains unchanged.

The learning smoke uses only the already passing21 TRAIN video/audio smoke events
and63 native questions, not either partial/full cache or validation features. Use
the exact five saved initial models from the verified transport smoke: C0 and
E0G0/E0G1/E1G0/E1G1. Freeze encoders by using their immutable cached features; all
existing readout parameters are eligible for optimization. No encoder load,
checkpoint selection, quality evaluation, audit or calibration.

Choose the same first32 unique TRAIN questions sorted by SHA256('17:'+question ID)
for every arm, grouped into8 updates of4 microbatches. Keep all original options;
use a deterministic complete permutation keyed by seed17/question ID and map the
original target into that order. AdamW lr0.001, betas0.9/0.999, eps1e-8, decay0.01;
mean CE over four examples, gradient norm clip1.0; FP32, seed17, no dropout.
This is a fixed8-update mechanics probe, not a learning-curve quality experiment
or evidence of a capacity ceiling. Do not rank arms by these TRAIN losses.

CPU tests must first prove permutation/target alignment, no repeated/validation
examples, correct accumulation normalization and matching initial arm tensors.
GPU smoke records every loss, gradient norm and module-group gradient reach, changed
parameter groups, effective counts, wall/step time and peak VRAM. The first zero
correction step can have zero upstream residual gradients; check reach on later
updates. Disabled modules and unused Text/Image projections must remain unchanged.
All losses/gradients/weights must remain finite. Save initial/final states, optimizer
state, fixed input plan, predictions and hashes in a new directory. Same-policy
reload must agree exactly. Fresh-process replay repeats from the initial state;
require exact deterministic histories/states/predictions apart from timing.

Use the idle local RTX3080 Laptop only; no upgrade/cloud. Reuse the proven native
training backend settings (deterministic algorithms, TF32 off, cuDNN benchmark off,
four CPU threads). Guards:15minutes,4GiB CUDA allocation,6GiB host peak,16GiB disk
reserve. Inspect GPU ownership before launch. Record concurrent CPU video caching.
No main learning run is admitted before the full video cache and joint binding
pass. Use measured smoke cost to choose a separately documented common full-run
budget and evaluation schedule; preserve all original quality/retention gates.

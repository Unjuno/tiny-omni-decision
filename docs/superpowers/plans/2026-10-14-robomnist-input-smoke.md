# Fixed RoboMNIST raw input smoke implementation plan

> **For agentic workers:** Use superpowers:executing-plans; implement inline.

**Goal:** Actually extract/decode one fixed permitted audio/video/CSI recording.
**Architecture:** Reuse approved source directory/HTTP reader; new pure validators,
small CPU-only producer and saved-product replay. No readout/model/payloadGT.
**Tech Stack:** Existing Python3.11/NumPy/PyAV/stdlib wave/zipfile, no upgrades.
**Spec:** `docs/pretrained_reuse/R2_ROBOMNIST_INPUT_PROTOCOL_20261014.md`.

## Global constraints

Exactly fixed Ar1_Vhigh_Ac0_Rp01_UNC062/7input files;3camera/3mic/CSI, no posGT.
64MiB reserved transfer/900s/384MiB outputs/1GiB disk/2GiB host. CPU only; preserve
old8GiB GPU guard and all legacy outputs. No retries/changed source/labels/fit/model.

## Review focus

Cross-group/substituted ZIP members must fail before payload opening.
No silent CSI timestamp cast, malformed complex eval or extra metadata features.
Decoder count/dimensions may differ from450nominal; report them, no dataset pruning.
Native PCM channels retained; sample/midframe policy fixed independently of quality.
Duration agreement alone cannot certify common-clock sync; replay binds actual files.

### Task1: Input validators and actual producer

Create `scripts/robomnist_input_arrays.py`, `scripts/fetch_robomnist_input_smoke.py`,
`tests/test_robomnist_input_arrays.py`. Pure functions
`parse_csi(value)->(arrays,metrics)` and `pcm_samples(raw,width,channels)`;
video decoder records all timing/frame digests and retains fixed samples.
RED→GREEN tests for finite/malformed CSI, timestamp/group corruption, PCM channel
preservation. Bound/pin7members; tests/Ruff/commit before actual execution.
Run new `C:/CodexArtifacts/pretrained-reuse-r2/robomnist-input-smoke-01` and observe.

### Task2: Saved raw/array replay and publication

New `scripts/replay_robomnist_input.py` plus regression tests: mandatory raw/NPZ
bindings, sample/channel/timestamp accounting, no partial/clock success laundering.
Reopen every permitted raw file, recompute PCM/CSI and all retained RGB arrays;
bitwise NPZ equality/hash/source/context. Record actual clock limitations/failures,
small report/README/ROADMAP, full CPU/Ruff/manifests, final reviewer, normal push/CI.
Overall full quality goal remains active; no new training or product selection.

# Real AV feature transport: verified before fitting

## What changed

Source `5499eeecf0765284a44da874f8a58ed1bfc46197`, managed branch
`codex/r2-av-transport`, based on published `8e90c80`. Both executing cache
checkouts were left unchanged. No old corpus, model, checkpoint or runtime
implementation was overwritten.

The old native FeatureStore creates relative sampled-order positions. A separate
`ExplicitEventStore` now preserves the saved actual event-clock video coordinates,
nominal20ms speech coordinates and event-audio center. It binds original questions
and every option to complete passing caches by event/parent/split/source digest,
line/index/order and clock. Full mode requires all frozen events. Smoke permits
only its declared TRAIN subset. Targets are not store inputs.

All streams stay FP32 with finite-shape checks and pinned files. Hashes are checked
at first file use and at the end; size/mtime changes are rejected while using the
bounded read-only memory maps. The store has no hidden pooling, synthetic time
replacement, option reduction or fabricated missing streams. Joint/audio/video
views use the same event. Complete candidate permutations are explicit.

18 new CPU tests cover incomplete/duplicate/mismatched bindings, clocks and source
identity, missing-input views, candidate ordering, exact positions, malformed
features, hash mismatches and files changed after a cached read. RED was observed
before implementation; all18 pass. Ruff initially found line-length issues, fixed
before execution. Full CPU suite2,293passed/1skipped, Ruff and25 manifest checks pass.

## Actually executed on saved native features

The21 previously reviewed TRAIN events yield63 unchanged material/action/reaction
questions. All24 original query vectors and the real25 speech/1 event/2048 video
tokens are loaded. No encoders are rerun here. Each question runs through:

- Original C0 Attention, encoder-inclusive196,254,874 parameters.
- Four residual arms E0G0/E0G1/E1G0/E1G1, exchange off/on × gate off/on, all
 197,097,949 allocated encoder-inclusive parameters.
- Joint, audio-only and video-only views for each arm.

Each residual arm uses the same initial tensors and C0 weights; policies differ.
The final correction layer is zero-initialized. E0G0 has extra capacity and is
not the original C0. These saved states are random readout mechanics fixtures,
not fitted candidates or promoted models.

| Measured item | First process | Fresh replay |
|---|---:|---:|
| Events / questions | 21 / 63 | 21 / 63 |
| Original decision records | 945 | 945 |
| Wall seconds | 100.0966 | 95.6463 |
| Prediction/check seconds | 88.9333 | 85.1286 |
| Peak process working set bytes | 732,246,016 | 732,942,336 |
| Candidate-permutation max logit difference | 0 | 0 |
| Optimizer updates | 0 | 0 |

Every record also checks reversed candidate order and same-policy checkpoint
reload. All original/reloaded outputs agree exactly; all initial residual outputs
equal C0; absent audio/video bypass the residual. Probability JSON sums to1 and
retains the same option association after reversal. The fresh process reproduces
all seven payloads exactly: five model states, binding metadata and prediction
JSONL. No label loss or Accuracy/NLL/Brier/ECE was computed from these random
readouts. Gate values are recorded only as mechanics diagnostics.

CPU FP32, one thread, seed17, deterministic PyTorch2.6.0+cu124, NumPy2.4.6,
Python3.11.9. CUDA hidden. Full audio GPU/video CPU caches were concurrent, so
these are not uncontended or end-to-end deployment timings. No cloud cost.

## Reproduction and saved evidence

Run from `C:/Users/junny/.codex/worktrees/r2-av-transport/tiny-omni-decision` with
the existing environment. Exact commands, source/input hashes and timestamps are
also saved in both result receipts:

```powershell
$env:PYTHONPATH='C:/CodexArtifacts/pretrained-reuse-r1/python-deps;C:/CodexArtifacts/pretrained-reuse-r2/native-speech-deps-01;C:/Users/junny/.codex/worktrees/r2-av-transport/tiny-omni-decision/src;C:/Users/junny/.codex/worktrees/r2-av-transport/tiny-omni-decision'
$env:CUDA_VISIBLE_DEVICES=''
$env:OMP_NUM_THREADS='1'
& 'C:/Users/junny/AppData/Local/Programs/Python/Python311/python.exe' -m scripts.probe_greatest_hits_transport --output C:/CodexArtifacts/pretrained-reuse-r2/greatest-hits-transport-smoke-01
& 'C:/Users/junny/AppData/Local/Programs/Python/Python311/python.exe' -m scripts.probe_greatest_hits_transport --output C:/CodexArtifacts/pretrained-reuse-r2/greatest-hits-transport-replay-01 --reference C:/CodexArtifacts/pretrained-reuse-r2/greatest-hits-transport-smoke-01 --reference-sha256 7f6b60c0d4dfa40cb9a86096301f6e5757a0e57ab6b3975b19d74ae4db8167d3
```

Both commands exited0. These outputs already exist; the runner rejects reuse.
No active run was stopped, restarted or changed.

Under `C:/CodexArtifacts/pretrained-reuse-r2/`:

| Receipt | SHA-256 |
|---|---|
| `greatest-hits-transport-smoke-01/result.json` | `7f6b60c0d4dfa40cb9a86096301f6e5757a0e57ab6b3975b19d74ae4db8167d3` |
| `greatest-hits-transport-replay-01/result.json` | `6c6bb69f2f7b2f62e8ae70467dc6f7243e80b23e39c85977f370f45fd0dec48e` |

Independent `greatest-hits-transport-verification-01/audit.py` SHA
`3fec9c576fcff88136c17387767c83b650fb499d922daf1219c5a70f2a7881c9` checks
all945 JSON decisions, coverage, probability reconstruction, replay and16 new
files. All8,808 earlier immutable files retain their hashes. Result status:
`PASS_REAL_FEATURE_TRANSPORT_REPLAY_AUDIT`. Large active caches are excluded from
this immutable ledger. The repository manifest
`manifests/pretrained_reuse/greatest-hits-transport-20261017.json` binds the audit.
Publication CI for the exact pushed commit is stored separately at
`greatest-hits-transport-verification-01/ci.json`.

## Still pending

Full video cache PID17848/session92865 and audio cache PID12824/session26274
continue with their original source/output locations. At an interim observation,
video1,868 and audio9,019 of11,151 events were saved. Require terminal success,
all events/questions/parents, and independent integrity audit before training;
these progress figures are not completion evidence.

Then freeze common initialization/data/order/optimizer/budget/validation rules for
C0 and the residual ablations. Include learning curves, native task and unimodal
controls, calibration metrics, four-modality retention and defensible paired
interventions. The Greatest Hits labels support scoped native-field development;
they do not certify both-required task truth, physical AV sync or unseen actor/site
quality. Current transport correctness does not establish gate benefit or the
four-modality95–98% product objective.

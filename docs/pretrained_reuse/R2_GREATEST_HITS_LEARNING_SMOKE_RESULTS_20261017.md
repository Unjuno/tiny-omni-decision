# Conditional AV readout: learning and replay measured

## Scope and execution

Execution source `304f28f568f0fe0dd3984ac23ed4ec2f7cd47578` on
`codex/r2-av-transport`. The [fixed protocol](R2_GREATEST_HITS_LEARNING_SMOKE_PROTOCOL_20261017.md)
was committed before either process. Full video extraction continues from its
original checkout, source and output; no running process was stopped or changed.
No Teacher, previous corpus or model artifact was overwritten.

The same passing 21 TRAIN events /63 native questions feed five arms:

| Arm | Added AV exchange | Added conditional gate | Encoder-inclusive parameters |
|---|---|---|---:|
| C0, original Attention | none | none | 196,254,874 |
| E0G0, residual control | off | off | 197,097,949 |
| E0G1 | off | on | 197,097,949 |
| E1G0 | on | off | 197,097,949 |
| E1G1 | on | on | 197,097,949 |

E0G0 controls additional residual capacity; it is distinct from C0. Disabled
components remain allocated in all four ablations. These are the exact random
readout initialization fixtures saved by the earlier transport probe, with
pretrained frozen encoder features. They are not warm starts from a fitted
production readout. The final residual layer starts at zero.

Each arm completes8 updates,4 microbatches/update, on the same32 distinct TRAIN
questions from18 events/assets. There are11 material,9 action and12 reaction
questions, no repeated question and no option removal. Target indices follow
complete candidate permutations. AdamW lr0.001, betas0.9/0.999, eps1e-8,
decay0.01, gradient clip1.0, FP32, seed17, constant LR for this smoke only.
All readout parameters are optimizer-eligible; encoders are not loaded or updated.

## What the actual GPU run established

Both processes exit0 with `PASS_AV_READOUT_LEARNING_SMOKE`:

- Finite loss, gradients and updated weights throughout all40 optimizer updates.
- Nonzero gradient reaches every enabled residual group, including both exchange
  directions and the gates. Initial zero correction can delay upstream gradients;
  reach is verified across all8 updates, not demanded at the first update.
- Unused Text/Image projections and disabled exchange/gate modules stay exactly
  unchanged. This does not prove overall unimodal retention: shared baseline
  parameters are trained and can affect other tasks.
- Each trained arm produces63 questions ×3 input views =189 saved decisions.
  Same-policy final checkpoint reload is exact for all945 decisions/process.
- With either modality missing, the residual bypass equals that checkpoint's own
  baseline exactly. It is not a claim of equality to the pretraining checkpoint.
- An independent process restarts from the same initial states and reproduces
  all22 payloads byte-for-byte: plan/config, five initial states, five final states
  including optimizer state, five histories and five prediction files.
- An independent CPU audit checks both runs'1,890 probability JSON records,
  softmax reconstruction, option association, training-plan alignment, optimizer
  step8 and preserved inactive parameters. All9,649 older protected files retain
  their hashes;46 new files are hashed.

Saved optimizer state is useful evidence, but this experiment tests restart from
initialization, not interruption/resume equivalence. No resume guarantee is made.

## Measured local cost

RTX3080 Laptop16GiB, Python3.11.9, PyTorch2.6.0+cu124, NumPy2.4.6, CUDA12.4,
FP32, deterministic algorithms, TF32 off, cuDNN benchmark off,4 CPU threads.
The existing automatic PyTorch SDPA policy is retained and recorded (flash,
memory-efficient, math and cuDNN enabled); a specific dispatched kernel was not
profiled. The CPU VideoPrism cache was concurrent. No dependencies were upgraded.

| Measured item | First process | Fresh replay |
|---|---:|---:|
| Total wall seconds | 42.5830 | 42.2235 |
| Input checks / loading seconds | 9.7126 | 10.1279 |
| Peak CUDA allocation bytes | 214,010,368 | 214,010,368 |
| Peak process working set bytes | 1,492,492,288 | 1,462,140,928 |
| Updates per arm | 8 | 8 |
| Unique questions per arm | 32 | 32 |

| Arm | First mean update seconds | Replay mean update seconds | First / replay median forward+backward seconds |
|---|---:|---:|---:|
| C0 | 0.2461 | 0.2159 | 0.1271 /0.1181 |
| E0G0 | 0.1970 | 0.1719 | 0.1725 /0.1572 |
| E0G1 | 0.2041 | 0.1743 | 0.1759 /0.1542 |
| E1G0 | 0.2041 | 0.2091 | 0.1763 /0.1733 |
| E1G1 | 0.2121 | 0.2078 | 0.1749 /0.1779 |

These are short, cached-feature measurements including checks and warmup/order
effects; C0 runs first. They do not establish that an added arm is faster than C0.
The204.1MiB CUDA allocation excludes resident encoders, and is not full-product
inference memory. Full-corpus I/O, validation and retention costs remain unmeasured.
Two processes total84.8065 wall seconds. Cloud spend0; local electricity unmeasured.

First process:2026-10-16T19:45:56.329396Z to19:46:38.846058Z.
Replay:2026-10-16T19:47:06.890208Z to19:47:49.059214Z.

## Explicit audio admission and tests

`load_revalidated_audio` verifies the independent receipt hash, exact failed
producer linkage, completed counts/input/feature checks, source root and every
saved artifact against its pinned ledger. It returns an in-memory cache-valid view
with original `INCOMPLETE` producer provenance. The original failed receipt stays
unchanged. Ordinary raw incomplete receipts still fail binding. This admission
path is executed in both smoke processes; training itself uses only the passing
21-event smoke banks. Full validation feature files are hashed for integrity but
zero validation examples are forwarded or scored.

18 new CPU tests cover admission failures, preserved producer bytes, deterministic
TRAIN-only sampling, complete permutations/target alignment and gradient
accumulation equivalence to mean loss. Initial tests failed before implementation;
the completed suite is2,315 passed,1 skipped. Ruff and25 existing manifest commands
pass. GPU work is outside normal CI. Exact pushed-commit CI is recorded separately.

## Reproduction and evidence

From `C:/Users/junny/.codex/worktrees/r2-av-transport/tiny-omni-decision`:

```powershell
$env:PYTHONPATH='C:/CodexArtifacts/pretrained-reuse-r1/python-deps;C:/CodexArtifacts/pretrained-reuse-r2/native-speech-deps-01;C:/Users/junny/.codex/worktrees/r2-av-transport/tiny-omni-decision/src;C:/Users/junny/.codex/worktrees/r2-av-transport/tiny-omni-decision'
$env:CUDA_VISIBLE_DEVICES='0'
$env:OMP_NUM_THREADS='4'
& 'C:/Users/junny/AppData/Local/Programs/Python/Python311/python.exe' -m scripts.probe_greatest_hits_learning --output C:/CodexArtifacts/pretrained-reuse-r2/greatest-hits-learning-smoke-01
& 'C:/Users/junny/AppData/Local/Programs/Python/Python311/python.exe' -m scripts.probe_greatest_hits_learning --output C:/CodexArtifacts/pretrained-reuse-r2/greatest-hits-learning-replay-01 --reference C:/CodexArtifacts/pretrained-reuse-r2/greatest-hits-learning-smoke-01 --reference-sha256 147621282f441751740ce3c3dedb4355cf4d450443449696a10f699e46b118a6
```

These directories now exist and must not be reused. Logs, CPU checks and the
independent audit are in `C:/CodexArtifacts/pretrained-reuse-r2/greatest-hits-learning-smoke-verification-01/`.

| Receipt under `C:/CodexArtifacts/pretrained-reuse-r2/` | SHA-256 |
|---|---|
| `greatest-hits-learning-smoke-01/result.json` | `147621282f441751740ce3c3dedb4355cf4d450443449696a10f699e46b118a6` |
| `greatest-hits-learning-replay-01/result.json` | `cd528908ba47e8bb95f06244d788bd5efea6e0886091a97d4b24af96f3f91ef2` |

Independent audit script SHA
`24170ab00e7e7ac470936980cff9c7dd4617dc14c6d3e187b38525f8c78483e0`;
status `PASS_AV_LEARNING_SMOKE_REPLAY_AUDIT`. Repository manifest:
`manifests/pretrained_reuse/greatest-hits-learning-smoke-20261017.json`.
The exact publication CI receipt is `greatest-hits-learning-smoke-verification-01/ci.json`.

## Next comparison and unresolved quality

The live full video cache had3,795/11,151 events and141/409 parents at the latest
publication observation. Do not treat progress or a partial bank as completion.
Require terminal success and full binding/audit before the main comparison.

Then fix the same initialization, training coverage/order, optimizer, sufficient
common budget and validation selector for C0 plus the four arms. Report full
learning curves and native material/action/reaction Accuracy/NLL/Brier/ECE, with
joint, unimodal, four-modality retention and paired input-change controls. Keep
diagnostic sensitivity separate from correctness when a swapped/conflicting input
has no defensible label. Gate values alone do not show useful complementarity.

These8-update smoke losses are not used for selection, accuracy claims, short-run
rejection, grokking or capacity-ceiling claims. No full quality comparison, calibrated
probability claim, both-required ground truth, new blind test, product promotion
or four-modality95–98% result has been established.

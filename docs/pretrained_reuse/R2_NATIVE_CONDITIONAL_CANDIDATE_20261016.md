# Native conditional candidate: actual integration, untrained

2026-10-16. **Functional integration PASS; task quality NOT EVALUATED.**
The new Path A candidate contains **196,254,874 parameters before quantization**.
It actually accepts native text, image, audio and video inputs, reuses their
tokens for supplied questions/candidates, and returns option probabilities as
JSON. Its 889,345-parameter connection/readout is initialized with seed 17 and
has received **zero optimizer updates**. No product promotion or accuracy claim
follows from this smoke.

[Machine-readable measurements](R2_NATIVE_CONDITIONAL_CANDIDATE_20261016.json) ·
[Fixed design](../superpowers/specs/2026-10-16-native-conditional-candidate-design.md) ·
[Implementation ledger](../superpowers/plans/2026-10-16-native-conditional-candidate.md) ·
[Strategy](NEXT_STRATEGY_20261016.md).

## What was executed

The same original pretrained encoders and initialized shared readout processed:

- Text: a literal observation about red cubes and blue spheres.
- Image: existing TRAIN Clevr-4 `CLEVR_new_000000.png`.
- Audio: existing verified TRAIN Lombard `s11_l_bbab2n`, decoded from the pinned ZIP.
- Video: the existing verified video of the same Lombard utterance.

This is **three native media payloads from two independent source groups**, plus
one authored text observation. Two fixed questions per modality produced eight
four-choice JSON outputs. The utterance's audio and video are not independent
examples. Questions are unlabeled plumbing checks; some have multiple plausible
answers. They must not be used as a single-target quality benchmark.

Native arrays retained by the same candidate:

| Input | Token shape | Preprocessing scope |
|---|---:|---|
| Text | 18 x 384 | Original DeBERTa token states |
| Speech | 110 x 384 | Valid 20 ms Whisper encoder tokens |
| Environmental audio | 1 x 384 | Existing EfficientAT clip representation |
| Image | 17 x 768 | Existing VideoPrism spatial image profile |
| Video | 2,048 x 768 | Existing eight-frame VideoPrism profile |

Each candidate/condition pair is encoded by the same original DeBERTa; the
readout attends to native token blocks. Arbitrary option IDs associate outputs
only and never enter the network. Coordinates preserve within-input sample/grid
order; they do not establish a common clock across independently sourced inputs.
No transcript decoder, correct-property mask or legacy learned semantic endpoint
was substituted for the new readout.

## Complete parameter accounting

| Resident component | Parameters |
|---|---:|
| One shared VideoPrism image/video tree | 114,365,184 |
| Original DeBERTa xsmall NLI model, including its unused classifier | 70,830,722 |
| Whisper encoder only | 8,208,384 |
| EfficientAT | 1,961,239 |
| Event mel preprocessor | 0 |
| New conditional readout | 889,345 |
| **Candidate total** | **196,254,874** |

Headroom under 200M is 3,745,126 parameters. Registered buffers add 12,199
elements / 51,088 bytes. FP32 parameter bytes are 785,019,496; parameters plus
buffers are 785,070,584 bytes. These are **tensor storage counts**, excluding
tokenizers, serialization and runtime overhead; they are not a packaged artifact
measurement. The <=200 MB deployment package gate is still open.

Save/reload verification temporarily holds a second readout with 889,345
parameters. Including that test helper, the live smoke contains 197,144,219
parameters. The visual worker remained live during the combined inventory and
Torch inference. The helper is not needed for product inference.

This is a new composition, distinct from the earlier 195,655,225-element
component inventory. It uses original DeBERTa weights and replaces the prior
task-specific readouts with the new untrained connection. Earlier accuracy
results do not transfer to it. Whisper encoder-only speech decision quality is
especially unproven; successful token extraction is not speech comprehension.

## Reproducibility and review

- Same-observation reuse, reversed candidate order and readout save/reload pass.
- A separate fresh process decoded inputs and recomputed all **21 feature,
  query and score arrays**: maximum absolute difference **0**.
- All eight JSON decisions reproduce their probability values exactly, with
  identical selected IDs. Original smoke-01 and revised smoke-02 arrays also
  match exactly.
- The report builder rechecked every output hash and **1,052 source/input pins**
  against actual files. Frozen Torch state hashes match before/after; the visual
  checkpoint, source and processing identities are pinned separately.
- Independent code review identified missing JSON-level replay comparison and
  staged-source detection. Both were fixed with regression tests in `4c4410a`.
  Whole-run sampled RSS and precise tensor-byte naming were added at the same
  time. The follow-up review found no remaining material integration findings.

Source inventories include validation **metadata**; validation media, reserved
calibration payloads and sealed audit were not loaded. Original encoders, legacy
Teacher/QAT/checkpoints, prior corpora and prior outputs were not overwritten.
The new runner shuts down only its own newly created visual worker.

## Execution resources

| Execution | Source commit | Wall seconds | Sampled process-tree peak RSS | CUDA peak allocated / reserved |
|---|---|---:|---:|---:|
| Initial smoke-01 | `64736c8` | 122.0005 | Encoding phase only; not a full-run peak | Preserved in original receipt |
| Revised smoke-02 | `4c4410a` | 15.7912 | 3,010,560,000 bytes | 486,964,224 / 505,413,632 bytes |
| Fresh replay-01 | `4c4410a` | 14.6924 | 3,016,499,200 bytes | 486,964,224 / 505,413,632 bytes |

RSS is sampled every 100 ms across the process tree, including the reload helper;
short peaks may be missed. Torch runs on the local RTX 3080 Laptop 16 GiB, with
FP32 weights, eager attention and TF32 disabled. VideoPrism runs in its existing
separate CPU JAX environment. Initial and subsequent wall times include setup,
hash checks and artifacts; this is not a steady-state throughput or p50/p95
benchmark. Total recorded inference-process wall time is 152.4841 seconds,
excluding preparation, tests and dependency inspection. GPU kernel-hours and
energy were not measured. No cloud compute or paid services were used.

Smoke-02 ran 2026-10-16 02:55:40–02:55:56 JST; replay ran
02:57:05–02:57:20 JST. Both exited normally. There were no training steps.

Preflight limitations are retained: an initially assumed image path did not
exist, so the actual existing full-coverage TRAIN path was located before the
config was finalized. A first invocation without dependency paths failed at
`soundfile` import before model loading or output creation. Existing dependency
directories supplied `soundfile`/`torchaudio`; nothing was installed or upgraded.
Smoke-01 predates JSON replay checks and whole-run RSS; use smoke-02/replay for
those claims. Its outputs remain preserved.

## Commands, revisions and artifacts

Execution source: `4c4410a4cd1cb906022868124cfba112157fb808`.
Config: `configs/pretrained_reuse/native_conditional_candidate_v0.json`.
Full model revisions, input pins, module origins and environment are in the JSON
report and original `result.json` files. Python 3.11.9, Torch 2.6.0+cu124,
torchaudio 2.6.0+cu124, Transformers 5.6.2, NumPy 2.4.6.

```powershell
$env:PYTHONPATH = 'C:/CodexArtifacts/pretrained-reuse-r1/python-deps;C:/CodexArtifacts/pretrained-reuse-r2/native-speech-deps-01;C:/CodexWorktrees/pretrained-reuse-omni/src;C:/CodexWorktrees/pretrained-reuse-omni'
$python = 'C:/Users/junny/AppData/Local/Programs/Python/Python311/python.exe'
# Historical commands: these output directories already exist and are protected.
& $python -u -m scripts.probe_native_candidate --output C:/CodexArtifacts/pretrained-reuse-r2/native-conditional-candidate-smoke-02
& $python -u -m scripts.probe_native_candidate --output C:/CodexArtifacts/pretrained-reuse-r2/native-conditional-candidate-replay-01 --replay-reference C:/CodexArtifacts/pretrained-reuse-r2/native-conditional-candidate-smoke-02 --replay-sha256 fc2251c2f845953791a0265f059f3b5b7cd15f9e2d5f4e4bb0d3081f1c477ed8
```

Run from the active worktree; any future reproduction must use a new output
directory. The runner rejects nonempty outputs and uncommitted source changes.

| Artifact | SHA-256 |
|---|---|
| Config | `f4d776b0137856b150a2e98f23ab8292924b4e4ace887536ecbd645056c2123d` |
| Smoke-02 `result.json` | `fc2251c2f845953791a0265f059f3b5b7cd15f9e2d5f4e4bb0d3081f1c477ed8` |
| Replay-01 `result.json` | `4c069a99f0807fa5b5f175884bdcaa9413236903aa2907c8556e438104805e3b` |
| Initialized `readout.pt` | `77a45ce55b5b17eb787ca649d2affbd6a8c72118f8953967eab41a02c2b2646b` |

Local directories under `C:/CodexArtifacts/pretrained-reuse-r2/`:

- `native-conditional-candidate-smoke-01`: original measurements and outputs.
- `native-conditional-candidate-smoke-02`: hardened runner outputs and readout.
- `native-conditional-candidate-replay-01`: fresh input-to-JSON reproduction.
- `native-conditional-candidate-verification-01`: full CPU/Ruff/manifest receipts.
- `native-conditional-candidate-report-01`: hash-checking builder and compact JSON.
- `native-conditional-candidate-publication-01`: repeated CPU checks and final
  publication receipt. Its setup error records an `args`/`command` receipt-key
  mismatch before checks; the corrected replay then passes all 27 commands.

## Verification and remaining gates

Ruff passes. CPU pytest: **1,910 passed, 1 skipped**; 24 targeted tests cover
the readout/input helpers. All **25 manifest CLI commands** pass. GPU/model work
was executed separately from CPU CI. Code commit `4c4410a` has successful
[CI run 1797](https://github.com/Unjuno/tiny-omni-decision/actions/runs/38781748712).
These checks establish implementation integrity, not task accuracy.

Still required: train the new connection using a frozen native task/data
contract; evaluate the same checkpoint on conditional text, image attributes/
count/relations, speech and environmental audio, temporal video and genuinely
joint decisions. Include source-held-out conditions, hard negatives, explicit
no-text task decisions, missing/shuffled inputs, time controls and retention.
Actual joint inference quality, optional spoken conditions, Japanese coverage,
calibration, compressed package, target-device latency and final audit remain
unverified. The overall R0–R8 objective stays active.

**Next action:** audit existing approved TRAIN/development sources and caches
against this exact input contract, freeze grouped tasks/selection/resource
limits, and initially train only the new readout. A measured learning curve
and task-by-task error attribution must justify any later encoder adaptation.
Do not restart closed prompt/loss/seed searches or quantize this untrained head.

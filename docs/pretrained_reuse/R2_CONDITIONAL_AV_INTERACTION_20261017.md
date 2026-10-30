# Conditional AV interaction: mechanics implemented, quality unmeasured

## Result

The user's proposed bidirectional exchange and conditional gates are now an opt-in
research readout. CPU contracts and the encoder-inclusive parameter calculation
pass. **No new model was trained; no accuracy improvement or adoption is claimed.**
The existing Attention baseline, trainers, corpus and checkpoints are unchanged.

Implementation and metadata audit source:
`e31cb35d5c56b99584e2b0146287c44d4766357e`.
Audit receipt SHA256:
`c56c73a13a30f6de4f71505e28e009343d1f84b6329d97711a2ada5677262b32`.

## Implemented path and comparison

```text
native blocks -> unchanged candidate Attention --------------------> logits
       |                                                            +
audio/video -> 8 latents each -> A<-V and V<-A in parallel             |
       -> candidate-conditioned readout -> two independent gates    |
       -> shared residual option scorer -----------------------------+
       -> existing probability JSON converter
```

The two gates depend on candidate/query and both observations. They can both be
high or both low; they are not an audio/video percentage split. They control only
the added residual path, not the baseline's contributions. They are neither
calibrated reliability scores nor causal explanations. Temperature1 JSON remains
uncalibrated pending separate calibration evidence.

The last residual layer is initialized to zero, preserving exact initial scores.
This initially delays gradients into the earlier added layers; tests explicitly
record this and separately check gradient reach with a disclosed nonzero-head
fixture. No fixture is labeled a learned checkpoint. Missing audio or video
bypasses the residual exactly. Text/Image continue through the existing baseline.

The comparison framework keeps the original C0 baseline plus four residual arms:
exchange off/on x gate off/on. These share initial tensors, pooling, scorer and
allocated size; the policy is stored in the checkpoint. Loading a different policy
fails before copying weights. This permits gate/exchange ablations; C0 alone would
confound added capacity with interaction. No arm has been fitted or ranked yet.

## Measured parameter ledger

| Component | Parameters |
|---|---:|
| Existing frozen encoder reference ledger | 195,365,529 |
| Existing native readout | 889,345 |
| Added AV modules | 843,075 |
| All readout modules allocated | 1,732,420 |
| Reference total before quantization | **197,097,949** |
| Remaining to200M | **2,902,051** |

Allocated modules count even when disabled in an ablation. The receipt's
`active_joint_parameters` means policy-enabled modules, not a measured nonzero
gradient count for an individual example; unused modality projections also remain
allocated. All instantiated modules are CPU/FP32. The existing encoder ledger is
reused; encoders were not loaded here. This is not an end-to-end RAM/VRAM or latency
measurement. Repeated baseline/residual projections cost compute, and 8-latent
pooling may lose detail. Full-field preprocessing and other failed policies are
not introduced into this module.

## Actual data prerequisite check

Read only the frozen common-corpus observation metadata, pinned SHA256
`e82febae93455cc06a2fb31f3dcc83ebcb79328552aff2ec4da507d660186afb`.

| Observation kind | TRAIN | Validation |
|---|---:|---:|
| Text | 2,626 | 1,756 |
| Image | 8,424 | 512 |
| Speech + event-audio features | 6,035 | 1,375 |
| Video | 256 | 200 |
| **Audio and video together** | **0** | **0** |

These are observations, not question or presentation counts. Two representations
of an audio source are not an audio/video pair. Training this AV-only residual on
the current common corpus would never activate it. Even positive AV presence
would not by itself establish synchronization, rights or complementary labels.
The audit therefore does not admit a joint training run.

Previous VIDIMU gates affected both inputs but failed useful-fusion retention,
even after a1,000-epoch curve. Recording matching and prior high correspondence
scores do not establish general semantic complementarity. Preserve those results.

## Next admissible comparison

First admit a rights-cleared paired task with source-grouped splits, measured
alignment and label evidence for audio-sufficient, video-sufficient and both-needed
questions. Preserve clean input evaluation alongside predetermined missing,
corrupted, irrelevant-swap, relevant-swap and time-offset conditions. A relevant
swap that changes the correct answer needs a verified replacement label. An
irrelevant swap should leave the answer stable. A prediction change alone is not
correct reasoning. Conflict handling needs task-defined truth/unknown semantics.

Then freeze a common data/order/seed/budget/loss/selector protocol for the arms,
run long enough to measure curves, and report joint gain, unimodal controls,
four-modality retention, NLL/Brier/ECE and resource costs. Gate statistics are
supporting diagnostics. Do not encourage equal gate usage or choose a model for
large swap sensitivity. The previously proposed long-budget unique-coverage
comparison remains a separate unfinished question, not combined with this change.

## Verification and artifacts

Targeted baseline + new model + metadata tests:36 passed. This covers initial and
missing-input parity, padding, candidate order, both-direction gradients, parallel
exchange inputs, independent conditional gates, same-policy exact save/reload,
cross-policy rejection, parameter budget and metadata fail-closed behavior.
Random CPU fixtures establish mechanics only. No media, validation labels, sealed
audit or GPU workload was loaded for this stage; optimizer updates0, cloud cost0.

Final local verification: Ruff PASS; full CPU suite2,193 passed/1skipped; all25
manifest validation commands PASS. Rehashing994 previously pinned artifact files
found0changes. The original user worktree's three preexisting changes remain
untouched. Baseline/trainer source diffs against673932c are empty. Publication CI
is checked against the pushed commit and recorded separately in local verification.

CI1851 initially found4metadata-test failures because importing the metadata helper
also imported optional PyTorch; the local ML environment concealed that dependency.
Moved the model import into the parameter-audit entry point and kept the metadata
schema independent, with a runtime schema-equality check. A fresh subprocess with
PyTorch explicitly unavailable reproduced RED and now passes GREEN. The model,
numerical logic and original audit receipt are unchanged. Failed CI/logs remain
recorded; no CI dependency or test was removed to hide the failure.

Local immutable artifacts:

- `C:/CodexArtifacts/pretrained-reuse-r2/conditional-av-interaction-contract-01/`:
  failing-first and passing test logs, final verification.
- `C:/CodexArtifacts/pretrained-reuse-r2/conditional-av-interaction-audit-01/`:
  command, source/environment hashes, four parameter ledgers and metadata counts.

Reproduce in the existing CPU environment:

```text
python -m pytest tests/test_native_av_interaction.py tests/test_native_av_interaction_audit.py tests/test_native_candidate.py -q
python -m scripts.audit_native_av_interaction --observations C:/CodexArtifacts/pretrained-reuse-r2/native-common-corpus-02/observations.json --observations-sha256 e82febae93455cc06a2fb31f3dcc83ebcb79328552aff2ec4da507d660186afb --output <new-directory>
```

Set `CUDA_VISIBLE_DEVICES` to the empty string; audit requires clean committed
source and refuses an existing output directory. The metadata script does not
read media or any split's question payload. Total product95–98%, joint/OOD,
calibration and deployment gates remain unmet.

Design: [conditional AV specification](../superpowers/specs/2026-10-17-conditional-av-interaction-design.md).
Prior evidence: [VIDIMU long gate curve](R2M_GATED_BUDGET_RESULTS_20261012.md),
[complementarity audit](R2_FUSION_COMPLEMENTARITY_20261012.md).
Research motivation: [GMU](https://arxiv.org/abs/1702.01992),
[Attention Bottlenecks](https://arxiv.org/abs/2107.00135); neither is reproduced here.

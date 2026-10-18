# Saved-prediction task attribution for text-free CLEVRER video (2026-10-18)

## Scope and provenance

This is a read-only attribution of the already selected predictions from the
[mean-MLP / BiLSTM probe](R2_TEXTFREE_VIDEO_TEMPORAL_RESULTS_20261018.md). It
does not run a model, select a checkpoint, tune a task, or inspect a sealed
audit. The selected checkpoints were chosen on this same 400-row development
validation set, so the new breakdown is descriptive and is not independent
validation evidence.

The auditor checks the pinned parent report, input preflight, frozen train and
validation manifests, selected prediction files, option probabilities, IDs,
targets, task metadata and source rows. It joins only the public typed program
and existing selected predictions. No natural-language question is fed into
the model. The report is outside Git at
`C:/CodexArtifacts/pretrained-reuse-r2/textfree-video-error-attribution-05/report.json`.

| Receipt | SHA-256 |
|---|---|
| Parent run report | `ebd35fb94cd655c982773e370200f987055a97a312536e9b3e0416a19d673365` |
| Selected mean-MLP predictions | `39171505753fae9e186bdb731ac3401b852470939a4e0cba05fdce144816a547` |
| Selected BiLSTM predictions | `2aad06613d16f942452f0e6cbfecd0f778c6bf42582e0db10aeb5d2cb7f71dab` |
| Frozen train manifest | `5cd4e5155b0161e35b10aad2c98e14afba7d46407e635bd1c08d786e1e51765d` |
| Frozen validation manifest | `53aa6f01447a43337a601279daf620c0df91d807773522cc92230fcab173db7c` |
| Attribution script | `dd67a7db442d0ef12ca303c20b294a0104fd51d7bbd1fef9a88443063b287de1` |
| Attribution report | `8d7930445cf3611af078a5158c4c485a6e59bbdc2a220aaddd64a04c1815eca8` |

Reproduce using existing artifacts:

```powershell
python -m scripts.audit_textfree_video_errors `
  --run-dir C:/CodexArtifacts/pretrained-reuse-r2/textfree-video-clevrer-typed-fit-01 `
  --output C:/CodexArtifacts/pretrained-reuse-r2/<new-attempt>/report.json
```

## Task and program coverage

The fixed validation contains 200 static and 200 temporal descriptive questions
from 200 scenes. Exact typed-program sequences are much less repeated for the
temporal subset:

| Type | Train examples / exact programs | Validation examples / exact programs | Validation examples with an exact program absent from train |
|---|---:|---:|---:|
| Static descriptive | 256 / 59 | 200 / 30 | 29 (14.5%); 6 unseen sequences |
| Temporal descriptive | 256 / 215 | 200 / 178 | 181 (90.5%); 164 unseen sequences |

All operators observed in validation also occur in training. The temporal
validation difficulty is therefore mostly novel **combinations/sequences**,
not an unseen operator vocabulary. Temporal training itself has 215 exact
programs in 256 examples, leaving few repetitions from which to learn those
compositions.

In `TextFreeVideoReadout`, each typed-program token has an embedding, but the
model averages those embeddings across the program before candidate scoring.
That representation does not retain operator positions or order. This is a
concrete task-encoding limitation to test in a later, separately preregistered
candidate; it is not proven to cause the observed errors. No distinct program
sequences with the same token multiset were found in these frozen train/validation
rows, so this audit does not demonstrate an actual label collision from that
invariance.

## Per-task selected-checkpoint results

`Majority` is the empirical most-common target rate in that validation cell,
reported only as a descriptive baseline. Balanced accuracy averages recall over
classes represented in that cell; support is shown because several cells are
small. The same validation set selected both checkpoints.

| Taxonomy / question type | n | Majority | MLP Acc / bal. Acc | BiLSTM Acc / bal. Acc | MLP NLL / Brier | BiLSTM NLL / Brier | BiLSTM wins / losses (net correct) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Count / static | 36 | 41.7% | 58.3% / 53.2% | 50.0% / 41.7% | 1.169 / .594 | 1.055 / .583 | 3 / 6 (−3) |
| Count / temporal | 56 | 32.1% | 25.0% / 19.0% | 26.8% / 20.4% | 1.637 / .807 | 1.568 / .779 | 4 / 3 (+1) |
| Exist / temporal | 48 | 54.2% | 47.9% / 46.0% | 52.1% / 49.5% | .782 / .577 | .705 / .509 | 3 / 1 (+2) |
| Color / static | 65 | 23.1% | 21.5% / 16.0% | 24.6% / 16.9% | 2.059 / .863 | 2.042 / .860 | 5 / 3 (+2) |
| Color / temporal | 42 | 19.0% | 14.3% / 15.5% | 9.5% / 11.2% | 2.242 / .899 | 2.154 / .889 | 0 / 2 (−2) |
| Material / static | 51 | 51.0% | 54.9% / 55.0% | 62.7% / 62.8% | .644 / .456 | .662 / .470 | 7 / 3 (+4) |
| Material / temporal | 25 | 52.0% | 68.0% / 68.3% | 72.0% / 72.1% | .628 / .432 | .598 / .408 | 3 / 2 (+1) |
| Shape / static | 48 | 37.5% | 64.6% / 62.7% | 54.2% / 50.4% | .915 / .515 | .935 / .552 | 0 / 5 (−5) |
| Shape / temporal | 29 | 48.3% | 44.8% / 41.1% | 48.3% / 39.3% | 1.081 / .650 | 1.039 / .622 | 4 / 3 (+1) |

The largest visible failure is class collapse. On temporal color, the BiLSTM
predicts `blue` 28/42 times although `blue` is the target only 3/42 times; it
gets 1/3 blue examples correct and has zero recall for brown, cyan, green, red
and yellow. On temporal count it predicts `1` 41/56 times while the target is
`1` only 16/56 times; it gets none of the 11 target-`3` or 2 target-`4`
examples correct. Some supports are too small for strong per-class conclusions.

The BiLSTM's net correct-example change versus MLP is +2 on temporal existence,
+1 count, +1 material, +1 shape, and −2 color; static shape loses 5 and static
count loses 3. This breakdown is consistent with the earlier aggregate result:
there is no broad temporal Accuracy gain. It also explains why macro Accuracy
alone can conceal severe failures in color/count.

## Decision

Do not promote the BiLSTM or train another candidate on this already selected
validation set. Preserve the recurrent comparison as inconclusive for temporal
generalization. The next justified video experiment should freeze a separate
task-encoding intervention or a more representative task mix before execution;
the strong temporal exact-program novelty and mean-pooled program representation
make task encoding/data composition a concrete hypothesis. It must use a new
selection protocol or independent validation identities and report class support,
balanced metrics, calibration and scene-level uncertainty. The 95–98% product
goal and multimodal quality gates remain unmet.

Implementation verification: four dedicated CPU tests pass; Ruff passes for the
new auditor and tests. No GPU, model inference, new feature extraction,
quantization, cloud cost, or sealed-audit access occurred.

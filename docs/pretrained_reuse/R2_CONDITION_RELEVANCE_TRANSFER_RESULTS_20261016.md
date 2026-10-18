# Fixed condition-relevance transfer: completed, no promotion

2026-10-16 JST. All three frozen weight identities completed the registered
transfer once. Terminal integrity and the independent CPU output audit pass.
This closes the matched CE/relevance comparison. The original validation-based
adoption gate remains **FAIL**; both selected checkpoints remain **step 0**.

## Measured result

Authored structured-state text, **zero native media assets**: 256 independent
source states, 16,384 questions, all 40 candidates. Every state contributes 64
questions across the four seen/new question × seen/new candidate cells. States
are disjoint from the registered development partitions; each cell has 4,096
questions. This is development transfer, not sealed audit or four-modality quality.

Overall ECE pools all questions; macro ECE averages the 16 condition/field slices.
They are deliberately reported separately. Lower NLL/Brier/ECE is better.

| Frozen role | Step | Accuracy | NLL | Brier | Overall ECE | Macro ECE | Mean confidence | Minimum of 16 slices |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Initial | 0 | 43.3167% | 2.517243 | .863086 | .325490 | .332486 | .107677 | 9.9609% |
| CE selected | 0 | 43.3167% | 2.517243 | .863086 | .325490 | .332486 | .107677 | 9.9609% |
| Relevance selected | 0 | 43.3167% | 2.517243 | .863086 | .325490 | .332486 | .107677 | 9.9609% |
| CE endpoint | 1024 | 93.7012% | .143224 | .073403 | .035843 | .036887 | .971646 | 50.0000% |
| Relevance endpoint | 1024 | 94.1650% | .128840 | .053872 | .019650 | .019799 | .953355 | 50.8789% |

Initial and both selected roles share exactly the same weight hash and inference
outputs. They are not three independently trained replications. All 16 slices
are equal-sized, so their macro Accuracy/NLL/Brier equal the overall values.

| Query / candidate wording | Initial Accuracy | CE endpoint | Relevance endpoint |
|---|---:|---:|---:|
| Seen / seen | 42.9688% | 100.0000% | 100.0000% |
| Seen / new | 49.7070% | 99.8047% | 99.9756% |
| New / seen | 38.4277% | 87.5000% | 88.9648% |
| New / new | 42.1631% | 87.5000% | 87.7197% |

Every condition's Accuracy/NLL/Brier/ECE/confidence, all 16 fields and all actual
query/option-form slices are preserved in the [JSON](R2_CONDITION_RELEVANCE_TRANSFER_RESULTS_20261016.json)
and [745-row CSV](R2_CONDITION_RELEVANCE_TRANSFER_METRICS_20261016.csv).
Seen query indices rotate by state; their actual slice counts are preserved,
not described as a full equal-count crossing of every absolute query index.

## The remaining failure is visible below the aggregate

The two unseen count-question forms differ sharply. For `q11`, both endpoints
answer every count question correctly under all four candidate forms. For `q10`,
CE is 0% under all four forms. Relevance is 0%, 3.5156%, 6.25% and 17.1875% for
`attribute`, `reported`, `key_value` and `sentence`, respectively; each contains
256 states. Thus even the minimum of the 16 condition/field slices hides a
**0% actual syntax slice**. No questions or candidate forms were dropped.

This is evidence of conditional wording-transfer weakness within this authored
task family. It is not a proof of a 200M model capacity ceiling, general counting
inability, or failure of the visual encoder: these inputs already supply the facts.
Do not use these observed templates for a rescue prompt/training sweep.

## Supplementary uncertainty, not a replacement adoption rule

After completion, paired differences were computed from saved predictions/logits,
with no model forward or fitting. Relevance endpoint minus CE endpoint:

| Overall metric | Difference | Descriptive 95% interval |
|---|---:|---:|
| Accuracy | +0.4639 percentage points | [+0.3357, +0.6042] pp |
| NLL | -.014384 | [-.016367, -.012504] |
| Brier | -.019531 | [-.020595, -.018491] |

Existing `paired_cluster_interval` resamples all 256 whole states, 10,000 draws,
seed 17, percentile interval with linear quantiles and question-weighted means.
All four cells' intervals are in JSON. ECE is a point estimate; no ECE interval
was computed. These supplementary intervals do not correct multiple comparisons,
checkpoint selection or the adaptive research history. They cover one training
seed and this state/template distribution, not arbitrary unseen tasks.

The small transfer gain and better endpoint calibration do not change the
predeclared primary validation result: both arms already saturated the new16
validation Accuracy at 100%, and neither retained all47 slices at a trained
scheduled checkpoint. The >=3pp primary gain and trained-retention gates fail.
Selected weights are unchanged; no product promotion or compression follows.

## Execution and independent verification

- Start: 2026-10-16 01:35:37.943989 JST; finish: 02:22:23.398501 JST.
- Process PID11760 exited normally. The source was unchanged during inference.
- 1,966,080 candidate-pair forwards; three distinct weights, five explicit roles.
- Effective fixed preprocessing: maximum70 pair tokens, zero truncation.
- Local RTX3080 Laptop16GB; FP32/eager, Python3.11.9, Torch2.6.0+cu124,
  CUDA12.4, Transformers5.6.2; exact package versions are in JSON.
- Transfer wall time2805.469s (46.758min); peak CUDA allocation466,447,360 bytes,
  reserved555,745,280; peak host1,931,882,496 bytes. This excludes prior fits/replay.
- Two main fits + exact TRAIN replay + transfer total12359.107s, about3.433h of
  recorded process wall time. This excludes earlier preparation/smokes and is
  not GPU-kernel time or a measured electricity cost. Cloud cost0.
- No optimizer update, calibration, sealed audit, checkpoint reselection or
  additional model evaluation occurred during output analysis.

The independent auditor first rejects nonterminal results, verifies frozen
identities and all output hashes, then recomputes predictions and probabilities
and every metric/slice from the saved scores using a scalar math implementation.
Maximum probability discrepancy6.67e-16; maximum metric discrepancy8.89e-16.
Old fit/source/corpus/checkpoint hashes were verified again; no artifacts were
overwritten. Large weights/media/checkpoints remain outside Git.

## Reproduction and artifacts

All local paths below begin `C:/CodexArtifacts/pretrained-reuse-r2/`.

| Evidence | Directory / SHA256 |
|---|---|
| Transfer | `condition-relevance-postfit-transfer-01` |
| Terminal result | `4949b62e6cdff034fbca05d55cf31f52872083ca3f7f216d4da7f3949be482c4` |
| Output audit | `condition-relevance-transfer-audit-01` |
| Audit result | `6467d66425f1819fc644c41452a32b51198f56aaa548f6a0c45c6f74d25aefdd` |
| Frozen comparison | `condition-relevance-postfit-comparison-02` |
| Comparison receipt | `b7541d4f7ec7b3736e78bd48d364f9fd007ffd985d9730e496ceb02ee5fa583e` |
| Report builder and supplementary analysis | `condition-relevance-transfer-report-01` |
| Source-level integration gaps | `condition-relevance-integration-readiness-01/result.json` |

Inference source `f2dc24872a22dad92344c2c3009c68f9feb5d44c`;
independent audit source `ff95b0f2b8cda187838cfa2e91a7766c7a2e7841`.
All weight identities and output hashes are in JSON. Original launch command:

```text
python -u -m scripts.evaluate_condition_relevance_transfer --comparison C:/CodexArtifacts/pretrained-reuse-r2/condition-relevance-postfit-comparison-02 --comparison-sha256 b7541d4f7ec7b3736e78bd48d364f9fd007ffd985d9730e496ceb02ee5fa583e --mode transfer --smoke-reference C:/CodexArtifacts/pretrained-reuse-r2/condition-relevance-postfit-train-replay-01 --smoke-sha256 ce12d2c33b7266702bb708938db67ea5c7ff91672a7eaf1f85d3952285b5c338 --output C:/CodexArtifacts/pretrained-reuse-r2/condition-relevance-postfit-transfer-01
```

This completed inference is not to be repeated for tuning. Saved-output audit:

```text
python -m scripts.audit_condition_relevance_transfer --directory C:/CodexArtifacts/pretrained-reuse-r2/condition-relevance-postfit-transfer-01 --result-sha256 4949b62e6cdff034fbca05d55cf31f52872083ca3f7f216d4da7f3949be482c4 --comparison C:/CodexArtifacts/pretrained-reuse-r2/condition-relevance-postfit-comparison-02 --comparison-sha256 b7541d4f7ec7b3736e78bd48d364f9fd007ffd985d9730e496ceb02ee5fa583e --output C:/CodexArtifacts/pretrained-reuse-r2/condition-relevance-transfer-audit-01
```

Use a fresh output directory for a future audit; the recorded directory is
immutable. The report's `build_report.py`, its hash and exact inputs are saved
locally; it uses the existing paired-cluster helper and performs no model forward.
Ruff,1886 CPU tests (one skipped), and25 manifest commands pass locally.

## Next action

Close this relevance intervention. Define one explicit native candidate and
task contract, preserving these diagnostic endpoints separately from selection.
Source inspection found that the old dispatcher loads MiniLM task runtimes,
the native transcript-choice scorer needs the Whisper decoder, and native AV
correspondence exposes fixed51-slot/four-video semantics. They cannot simply be
relabeled as the195.66M general conditional composition. Reuse their verified
components while accounting for the real connections and full parameter budget.

Independent native/temporal/joint/OOD quality, all required tasks >=95%, exact
integrated <=200M parameters, package and latency gates remain open. Follow the
[current strategy](NEXT_STRATEGY_20261016.md); no further synthetic rescue sweep.

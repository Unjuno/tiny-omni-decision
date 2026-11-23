# R2 BERT-family evidence reconciliation — 2026-11-24

## Decision

BERT-family encoders are worth retaining as **task-specific text references and possible teachers**. Current evidence does not justify another broad architecture sweep, replacing the multimodal stack with a BERT-base, or claiming the product's text target is solved. The larger BERT-base repeatedly improves selected text tasks but exceeds the provisional four-tower parameter budget. Compact Japanese DistilBERT fits the projected budget in one configuration, but its measured results vary substantially by task and remain far below the 95–98% aspiration.

The next useful step is not another run on an exposed validation or public pseudo-test. It is to secure one source-disjoint, rights-admitted decision task and compare a single frozen compact BERT-family reference with the existing text representation using the same option scorer and frozen split. If no such source is admitted, stop adding BERT experiments and use existing results to inform a future teacher-distillation design.

## Newly reconciled evidence

The following results are recorded on separate remote experiment branches, not in this source-screen commit's tree. Their result reports and external artifacts remain authoritative; this summary does not reproduce or select their models.

| Task / provenance | BERT-family result | Interpretation and limit |
|---|---|---|
| Jamp temporal NLI, publisher test; branch `codex/r2-jamp-bert-temporal-probe-20261124`, commit `b3dee21b42e859fe6941698d56b944343a52d188` | Frozen Tohoku BERT-base: 56.32% accuracy, 0.9522 NLL; first-four-layer LINE DistilBERT with top-two layers adapted: 56.90%, 0.8956 NLL. | A narrow template-generated temporal-text task. Test templates overlap development templates, so it does not estimate unseen-template generalization. The adapted run's validation-selected checkpoint was not best on the test; do not promote it from this result. The source declares CC BY-SA 4.0 and references a separately licensed case-frame dictionary; check project admission before any reuse. |
| JAQKET Japanese 20-option quiz, official Dev2 pseudo-test; branch `codex/r2-jaqket-choice-bench-20261124`, commit `f5b4cbaca632085f61177e62729e3ba1d3430985` | Frozen Tohoku BERT-base pair scorer: 38.42% accuracy, 2.069 NLL; first-four-layer LINE DistilBERT: 24.57%, 2.495 NLL; chance is 5%. | BERT-base used question context and beat its option-title-only control (21.16%). Dev2 was evaluated once after freezing; target entities still overlap train and distractors. This is a public pseudo-test, not blind evidence. It is knowledge-intensive text-only QA, not broad world understanding. |
| Earlier Japanese typed-decision publisher test | Tohoku BERT-base: 52.54% macro teacher-mode accuracy, 1.0639 NLL; compact 27.6M LightSPLADE-derived encoder: 44.17%, 1.1299. | Targets are synthetic teacher distributions; 31% of candidate pairs were truncated. Supports BERT as a teacher/reference only. |
| Earlier JCoLA OOD development comparison | Frozen BERT-base and first-four-layer DistilBERT both predicted the majority class for every record; 73.28% sample accuracy and 0% unacceptable recall. | Rejects that frozen shared-scorer setup for this task; does not establish a BERT-family ceiling. This development split is exposed and must not be tuned against again. |

For detailed protocols, caveats, and hashes, see the reports on those branches: `R2_JAMP_BERT_TEMPORAL_PROBE_20261124.md`, `R2_JAQKET_BERT_PROBE_20261124.md`, `R2_TYPED_DECISIONS_JA_PUBLISHER_TEST_AUDIT_20261123.md`, and `R2_JCOLA_BERT_GROUPED_RESULTS_20261123.md`.

## Model-size implication

The pinned four-tower estimate is approximately 253.29M parameters when Tohoku BERT-base replaces the existing Japanese text encoder, so it exceeds the provisional 200M **pre-quantization parameter** target. A first-four-layer LINE DistilBERT proposal is approximately 195.99M in that stack, leaving little parameter headroom. These are parameter projections, not measurements of the final quantized package, resident runtime memory, or deployment latency. The compact configuration's task results are not strong enough to justify accepting that narrow headroom yet.

## Integrity and rights boundaries

- Do not retune or repeat evaluations on the now-exposed Social IQa publisher dev, MASSIVE validation/test, typed-decision publisher test, Jamp test, or JAQKET Dev2.
- JAQKET's developer-created Dev1/Dev2 files were screened separately from its non-commercial competition train and Wikipedia article corpus. The experiment does not authorize use of those excluded sources.
- Jamp's source is reported as CC BY-SA 4.0 and includes third-party-origin content conditions. Do not treat its result as blanket approval for future ingestion.
- KUCI and JamC-QA-V2 remain outside the current explicit license allowlist pending policy review; no payload was downloaded in the source screen.
- The GPQA HF gate includes an additional promise not to publish examples online. No access or evaluation should occur without explicit approval of that specific condition.
- No sealed audit was opened for this reconciliation, and no model was trained or evaluated in this documentation-only step.

## Next decision gate

Proceed with one compact BERT-family reference only if a rights-admitted, source-disjoint task with a frozen validation split is available. Predeclare the model, identical scorer, option-order controls, trainable parameters, stopping rule, and validation metrics before fitting. Require task-level calibration and leakage controls; do not use aggregate accuracy alone. If the task source is not admitted, preserve BERT-base as a teacher/reference and defer distillation until the task/data pipeline is ready.

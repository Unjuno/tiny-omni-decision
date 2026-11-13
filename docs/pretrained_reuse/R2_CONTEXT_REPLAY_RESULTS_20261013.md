# Fixed original-task replay result

The predeclared 384-update run completed. Seed 17; 3,072 unique questions: 1,536 context and 1,536 original training questions. Same coverage parent, fresh optimizer, all six encoder blocks plus classifier (10,647,169 trainable / 22,713,601 total parameters). Context exposure is halved relative to context-only training; this tests budget allocation, not adding replay at fixed context exposure.

Command: `python -m scripts.train_context_replay`; source commit `7ccd5de`. Full config, ordered IDs, predictions at all evaluation points, training CE/gradient norms, initial/final weights and hashes are saved in `C:\CodexArtifacts\pretrained-reuse-r2\context-replay-context-run-01-all-encoder`.

| Step | Plain Accuracy | Irrelevant Accuracy | Necessary Accuracy | Original Accuracy | Original NLL | Original Brier | Original ECE |
|---|---|---|---|---|---|---|---|
| 0 | 0.906250 | 0.509115 | 0.890625 | 0.635827 | 0.835591 | 0.470606 | 0.018915 |
| 128 | 0.920573 | 0.906250 | 0.907552 | 0.633071 | 1.091454 | 0.528862 | 0.148277 |
| 256 | 0.923177 | 0.907552 | 0.919271 | 0.640945 | 1.020091 | 0.514156 | 0.139940 |
| 384 | 0.934896 | 0.928385 | 0.923177 | 0.624803 | 1.124413 | 0.544218 | 0.159408 |

## Fixed final endpoint

| Condition | N | Accuracy | NLL | Brier | ECE |
|---|---|---|---|---|---|
| plain | 768 | 0.934896 | 0.168960 | 0.096352 | 0.026329 |
| irrelevant | 768 | 0.928385 | 0.163548 | 0.101430 | 0.013845 |
| necessary | 768 | 0.923177 | 0.171174 | 0.104595 | 0.021512 |
| original | 2540 | 0.624803 | 1.124413 | 0.544218 | 0.159408 |

Original retention at final is worse than the parent (Accuracy 0.635827, NLL 0.835591) and context-only final (Accuracy 0.628346, NLL 0.876646). Replay did not preserve original-task calibration: ECE rose from 0.018915 to 0.159408. Step 256 Accuracy improved slightly but NLL/Brier/ECE were already worse; do not retrospectively select it. Context-only final Accuracy was 0.937500/0.932292/0.936198 for plain/irrelevant/necessary; replay final remains close but does not solve the retention trade-off.

Measured timed section: 294.50 seconds (excludes startup); peak allocated GPU memory: 3,350,991,872 bytes. Local GPU only. Artifact hashes verified after completion. Exact same-instance saved-weight reload and frozen-parameter checks passed. Historical state-only truncation at 512 tokens retained; no example exclusions.

Single-seed development results, not blind evaluation or a multimodal quality claim. No product promotion; no sealed audit, compression or changes to existing artifacts. Further work needs to identify why original-task confidence worsens before another training sweep. The overall product objective remains unmet.

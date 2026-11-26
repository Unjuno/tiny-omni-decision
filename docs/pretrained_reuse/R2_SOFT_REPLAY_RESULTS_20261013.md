# Fixed soft-teacher replay result

384 updates /3072 unique examples completed; same seed17, initialization, order, optimizer and evaluation as hard replay. Only original training targets use normalized stored teacher distributions. Context targets remain hard. Fixed final endpoint; no checkpoint search.

| Condition | Hard Accuracy | Soft Accuracy | Hard NLL | Soft NLL | Hard Brier | Soft Brier | Hard ECE | Soft ECE |
|---|---|---|---|---|---|---|---|---|
| plain | 0.934896 | 0.929688 | 0.168960 | 0.169583 | 0.096352 | 0.095875 | 0.026329 | 0.030323 |
| irrelevant | 0.928385 | 0.930990 | 0.163548 | 0.163131 | 0.101430 | 0.101771 | 0.013845 | 0.020765 |
| necessary | 0.923177 | 0.938802 | 0.171174 | 0.144267 | 0.104595 | 0.085074 | 0.021512 | 0.020213 |
| original | 0.624803 | 0.639370 | 1.124413 | 0.859784 | 0.544218 | 0.478895 | 0.159408 | 0.030329 |

## Learning curve (soft targets)

| Step | Plain Accuracy | Irrelevant Accuracy | Necessary Accuracy | Original Accuracy | Original NLL |
|---|---|---|---|---|---|
| 0 | 0.906250 | 0.509115 | 0.890625 | 0.635827 | 0.835591 |
| 128 | 0.917969 | 0.906250 | 0.914062 | 0.635827 | 0.861046 |
| 256 | 0.928385 | 0.920573 | 0.925781 | 0.649606 | 0.844611 |
| 384 | 0.929688 | 0.930990 | 0.938802 | 0.639370 | 0.859784 |

Timed section 303.80s; peak allocated VRAM 3350995456 bytes. Startup excluded. All artifact hashes checked; initial weights and predictions byte-identical to hard replay. Evaluation IDs/options/targets match at every step. Frozen check and exact same-instance reload passed.

Output: `C:\CodexArtifacts\pretrained-reuse-r2\context-soft-replay-context-run-01-all-encoder`. Command: `python -m scripts.train_context_soft_replay`. Source `ae24b0a`. CPU suite730 passed/1 skipped; Ruff and manifest validation passed. Source commit GitHub Actions38364825176 and38364815885 succeeded.

This is single-seed reused development evidence. No blind audit, promotion, compression, or modifications of existing artifacts. Teacher distributions are synthetic annotations, not calibrated ground truth. Overall multimodal goal remains unmet.

## Interpretation

Soft replay improves original-task Accuracy/NLL/Brier/ECE versus hard replay.
Against the original coverage parent (.635827 Accuracy/.835591 NLL/.470606 Brier/
.018915 ECE), final soft replay is .639370/.859784/.478895/.030329: slightly higher
Accuracy but worse proper scoring and ECE. Do not claim complete retention.
Context final Accuracy is92.97/93.10/93.88%, below the95–98% aspiration and limited
to this synthetic condition task. This one-factor result supports target sharpness
as a contributor to hard-replay overconfidence in this run, not its only cause.
No seed replication or paired uncertainty for this contrast yet; no product adoption.

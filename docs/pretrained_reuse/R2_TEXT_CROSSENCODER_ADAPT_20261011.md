# Approved final-block Cross-Encoder adaptation

The user approved the bounded experiment on2026-10-11. Source is the pinned
22,713,601-parameter MS MARCO MiniLM-L6 model. Update exactly
bert.encoder.layer.5.* and classifier.*:1,774,849 parameters, no new model blocks.
All other parameters are frozen and hashed before/after every epoch.

Predefined run: seed17,3epochs,8questions per update, AdamW2e-5,
weight decay .01, constant LR, gradient clipping1.0. Dropout disabled in all
modules. Same frozen train23,319/validation2,540 IDs with binary criteria retained;
question/candidate first, state second,512tokens, state-only truncation. Reject
question/candidate overflow. Each question contributes equal CE regardless of
option count. Select earliest minimum validation NLL. Save best/final deltas
separately. Reload best and require exact saved-validation-logit parity. No
calibration fitting, new seeds, sealed audit or product promotion.

CPU suite316passed,1skipped; Ruff passed. Separate real-model smoke used16train
and8validation examples,2updates: frozen hash unchanged, nonzero selected
gradients, no frozen gradients, best reload/repeat parity PASS, peak allocated
VRAM410,522,624bytes. Smoke metrics are not quality evidence. Its result remains
C:/CodexArtifacts/pretrained-reuse-r2/text-crossencoder-adapt-smoke-01/result.json.

Main output: C:/CodexArtifacts/pretrained-reuse-r2/text-crossencoder-adapt-v1.
Runner: scripts/train_text_crossencoder.py. Inputs match the smoke command with
--smoke omitted. Main process is separately seeded; smoke state is not reused.
No exact optimizer resume is implemented. A stopped run would be incomplete.

## Completed result (2026-10-12 JST)

Execution PASS: process exited 0 after 3 epochs / 8,745 optimizer updates.
Source commit: a98df3a. Exact command, effective configuration, input hashes,
history and artifact inventory are in the adjacent JSON report.
23,319 unique train questions / 6,682 states were consumed 3 times:
69,957 total question consumptions. Selected epoch 3 has the same consumption
count as the full run. Validation contains the same 2,540 questions / 732 states.

| Epoch | Update | Rolling train CE | Validation Accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 2915 | 1.00827 | 54.57% | 0.97317 | 0.55381 | 0.04225 |
| 2 | 5830 | 0.95254 | 56.77% | 0.94614 | 0.53885 | 0.03624 |
| 3, selected | 8745 | 0.93446 | 58.11% | 0.93779 | 0.53431 | 0.04276 |

| Same validation IDs | Accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|
| Criteria-preserving frozen L3 representation + readout | 50.91% | 1.01490 | 0.58347 | 0.03778 |
| L6 Cross-Encoder zero-shot | 45.98% | 1.19869 | 0.64975 | 0.13666 |
| L6 final-block adaptation | 58.11% | 0.93779 | 0.53431 | 0.04276 |

The adapted model improves Accuracy by 12.13 percentage points versus the same
zero-shot L6, and 7.20 points versus the frozen L3 readout. The L3 comparison
changes pretraining, architecture and input processing as well as adaptation;
it does not isolate a pure architecture effect. All metrics are development
validation results on synthetic English data, with checkpoint selection on NLL.
They are not blind or four-modality product-quality evidence.

Train CE and validation NLL both improve across these three epochs; a plateau
or NLL overfitting is not established. ECE worsens from epoch 2 to 3, and remains
worse than the frozen L3 reference. Rolling train CE is not a fixed-example
train/validation gap. No extra seed or longer training was run.

Frozen parameter hashes match after every epoch. Reloaded best logits exactly
match their saved validation values and repeat inference. Best and final delta
files are separate; tensor equality was independently checked (both epoch 3),
with finite 1,774,849 trainable parameters. File SHA differs because serialization
containers differ. Best SHA-256:
`5fa8323a47e72c3796d0da35618b603fa49784cacb670dee0e910e8d5757ed78`.
Each delta is approximately 7.11 MB; it replaces existing parameters when applied,
adding no parameters. The underlying model remains 22,713,601 parameters.
No quantized deployment size or latency is claimed.

Local RTX 3080 Laptop: 1,311.22 seconds for training, scheduled validation and
reload checks, excluding model loading/tokenization. Separate evaluation timing
was not instrumented. Peak PyTorch allocated VRAM 834,455,552 bytes; this excludes
CUDA context/other allocator overhead. Process started 2026-10-11 23:38:03 JST;
result saved 2026-10-12 00:00:35 JST. No cloud compute or charges.
Input diagnostics retain all questions/candidates; state truncation affected
3,446 training and 239 validation candidate pairs.

Final verification: Ruff PASS, CPU tests 316 passed / 1 skipped. Legacy Teacher,
QAT and other run artifacts were not changed. Sealed audit was not opened.
The approved bounded experiment is complete; the broader Omni goal remains open.
Next quality evidence needs unseen criteria/task and multimodal conditional
reasoning checks before product selection. Do not infer deployment readiness
from this text-only gain.

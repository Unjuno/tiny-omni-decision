# Fixed mixed-corpus repair pilot

Execution PASS:512updates completed; frozen hashes and exact best-logit reload
passed. Starting point is saved adapted epoch3; optimizer is new AdamW, not
exact training resume. Same final-block/classifier1,774,849 trainable parameters.
Smoke was a separate process with2updates/16training questions and8validation
questions per source; it passed before main launch. No old artifact overwritten.

| Step | Original Acc | Original NLL | Curriculum Acc | Curriculum NLL | Selector |
|---|---:|---:|---:|---:|---:|
| 0 | 0.58110 | 0.93779 | 0.23389 | 1.45684 | 1.19731 |
| 128 | 0.58346 | 0.93604 | 0.23535 | 1.45747 | 1.19676 |
| 256 | 0.57598 | 0.95632 | 0.24414 | 1.45710 | 1.20671 |
| 384 | 0.58346 | 0.93910 | 0.23584 | 1.45227 | 1.19568 |
| 512 | 0.57835 | 0.93659 | 0.24365 | 1.44972 | 1.19316 |

Best is512 by the predefined equal-corpus mean NLL. Final and best are separate
files with equal tensors. Original Accuracy falls0.28percentage points while
NLL/Brier/ECE slightly improve. Curriculum Accuracy rises0.98points but stays
near uniform expected23.75%; uniform NLL1.44208/Brier0.76250 remain lower than
the final1.44972/0.76658. No useful conditional-reasoning gain is established.

All18 selected tensors changed from initialization; no nonfinite values.
Consumed4,096 unique questions:2,048 per source, no question repeats. Curriculum
coverage is25% of8,192 available training questions. Initial full original
validation reproduces the saved reference within1e-6 on all four metrics.
All4518 validation questions are evaluated at0/128/256/384/512. Task counts,
unique-state counts, per-task metrics and each prediction are retained.

Cumulative rolling train CE rose1.1676->1.1842 across changing batches; it is
not a fixed-sample gap. A512-update pilot covering25% of the new curriculum
cannot establish a clean capacity ceiling. Validation evidence remains reused
development data, not general/4-modality/95% performance or a final audit.

Local RTX3080 Laptop, FP32. Timed training/validation/reload253.64sec, peak
PyTorch allocation801,630,208bytes. Loading/tokenization excluded; per-evaluation
seconds are in history. No cloud costs. Base/initial/data/config/prediction
hashes and command are in JSON. No exact optimizer/RNG resume is supported.

Source commit0f91426. Ruff PASS; CPU suite323passed,1skipped. No extension,
new seed, architecture change or further training was launched after the pilot.
Any next budget experiment must be declared separately and start from the same
preserved initialization if it is intended to isolate coverage/budget. Do not
treat a negative pilot as evidence that more training necessarily helps.

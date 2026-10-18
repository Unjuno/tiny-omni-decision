# Approved fixed 2x2 condition diagnostic

All64 states,256 questions, targets and ordering are byte-identical to the
original approved probe. Only weights (base/adapted) and candidate text
(`A` / `A: Candidate A.`) vary. No numerical facts are added to candidates.
All four conditions were run once in separate processes; no training or selection.

| Weights | Candidate | Accuracy | NLL | Brier | ECE | Both correct /128 | Changed /128 |
|---|---|---:|---:|---:|---:|---:|---:|
| adapted | bare | 0.35938 | 1.15890 | 0.70772 | 0.15749 | 0 | 0 |
| adapted | described | 0.35938 | 1.16762 | 0.71752 | 0.18237 | 0 | 0 |
| base | bare | 0.35938 | 1.30977 | 0.77606 | 0.24980 | 0 | 0 |
| base | described | 0.35938 | 1.19687 | 0.73390 | 0.21622 | 0 | 0 |

All four conditions select B for every question. The selected adapted/bare
replay matches original probabilities with max error 0.
The minimal candidate description does not restore condition tracking, and
the same argmax failure already exists in the pretrained model. Thus this
particular failure was not introduced solely by final-block adaptation.
Adaptation changes confidence/calibration metrics without fixing the decisions.

This excludes only this narrow serialization repair and this adaptation-only
explanation. It does not prove a general architecture ceiling, nor that richer
candidate descriptions cannot help. Do not search further wording on this probe
or treat its results as blind product evaluation. No product promotion follows.

Local RTX3080 Laptop, FP32, PyTorch2.6.0+cu124; each process loads the pinned
22.7M base, optionally overlays the same selected delta, and performs768 pair
scores. Exact runtime, peak allocation, revision, input/runner/output hashes
and per-condition subgroup metrics are in the adjacent JSON. No cloud costs,
sealed access, legacy modifications or additional model training.

Source commit261c108. Ruff and CPU tests are checked with this record.
The text-quality gate remains unmet; further capacity or training changes need
a separately justified task/architecture plan, not repeated tuning of this test.

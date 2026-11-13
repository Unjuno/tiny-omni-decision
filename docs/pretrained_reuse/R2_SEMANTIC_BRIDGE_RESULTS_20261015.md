# Frozen semantic-state bridge: condition binding is weak on this probe

**Executed, inference-only, TRAIN-only.** All256 fixed images /1,024questions
completed; the fresh process reproduced22 saved payload files exactly. No fit,
prompt search, calibration, new checkpoint, validation/audit access or product
adoption. The <=200M four-modality95–98% goal remains unmet.

[Prospective protocol](R2_SEMANTIC_BRIDGE_PROTOCOL_20261015.md) ·
[Full metrics, hashes and accounting](R2_SEMANTIC_BRIDGE_RESULTS_20261015.json).

## What this changes

The retained text Decision model fails to reliably bind a question to the right
field **even when given correct facts**. All1,024 oracle predictions are values
present in the supplied state, and within the correct field's10options accuracy
is100%. But choosing across the same40typed options gives only277/1,024correct
(27.05%). All-four accuracy is0/256; changed-question pairs both-correct34/1,536.
Only99/256images have any argmax change across the four questions.

This makes conditional field binding/semantic transfer the next priority before
another image-encoder or readout-capacity change. It does not prove universal
language or encoder incapacity: serialization, typed descriptions and40choice
transfer are new to this text model. Native Top1facts discard uncertainty, and
the image head trained on these images. This diagnosis is not a blind quality
result, a replacement benchmark, or evidence that the complete product is95%.

## Same40choices, four questions per image

All conditions keep all256images and all40typed options. Brier is the sum over
classes; NLL is stable log-softmax without probability clipping; ECE has15bins.
All four questions have equal counts, so overall and macro Accuracy coincide.
Macro ECE differs from pooled ECE and is retained in the JSON.

| Input condition | Accuracy | NLL | Brier | Pooled ECE | Mean confidence |
|---|---:|---:|---:|---:|---:|
| Correct annotation facts (oracle) | .270508 | 2.259885 | 1.002459 | .347977 | .617000 |
| Frozen image head's predicted facts | .239258 | 2.408755 | 1.033273 | .374940 | .614198 |
| No evidence | .051758 | 3.675709 | .974400 | .003627 | .048131 |
| Oracle facts, no question | .250000 | 2.804548 | 1.163759 | .512349 | .762349 |
| Next image's oracle state, original labels | .044922 | 8.229510 | 1.407539 | .572078 | .617000 |
| Same donor state, donor-corrected labels | .270508 | 2.259885 | 1.002459 | .347977 | .617000 |

Low pooled ECE in the no-evidence condition accompanies5.18% Accuracy; it is not
high-quality calibrated decision making. Donor scores are reused exactly from
the already evaluated donor inputs. There are no extra donor forward calls.
The question-free25% is the maximum possible when one prediction is reused for
four distinct typed targets. Uniform40choice chance is2.5%, not25%.

| Question | Oracle Accuracy | Predicted-state Accuracy | Oracle NLL | Predicted-state NLL |
|---|---:|---:|---:|---:|
| Color | .566406 | .578125 | 1.048599 | 1.028022 |
| Texture | .109375 | .113281 | 2.233055 | 2.336426 |
| Count | .398438 | .257812 | 2.248354 | 2.795296 |
| Shape | .007813 | .007813 | 3.509531 | 3.475276 |

Oracle selected-field counts; rows are requested field, columns are selected
color/texture/count/shape. This distinguishes fact matching from question binding.

| Requested field | Color | Texture | Count | Shape |
|---|---:|---:|---:|---:|
| Color | 145 | 7 | 99 | 5 |
| Texture | 100 | 28 | 113 | 15 |
| Count | 135 | 13 | 102 | 6 |
| Shape | 78 | 23 | 153 | 2 |

Predicted-state within-correct-field accuracy is94.2383%, equal to the native
head's965/1,024correct facts. Its40choice field-routing accuracy is26.9531%, with
245/1,024correct answers. The59wrong native facts explain some losses, but correct
perception does not ensure successful conditional selection:714questions have
correct native facts and fail in both oracle and predicted-state40choice arms.
Other-field perturbations can also change ranking; the JSON keeps all eight
perception/oracle/predicted correctness intersections.

## Original native10choice scores, separately

These are TRAIN diagnostic results from the retained image head, not new held-out
scores and not directly comparable to40choice Accuracy above.

| Task | Accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|
| Color | 1.000000 | .001485 | .000056 | .001468 |
| Texture | .980469 | .035242 | .018288 | .012089 |
| Count | .789063 | .611530 | .337082 | .129236 |
| Shape | 1.000000 | .001276 | .000127 | .001234 |

## Reproduction, inputs and accounting

Source commit `9be76c9496d52ebe4475b1034cd11c7278d6627c`.
Output root `C:/CodexArtifacts/pretrained-reuse-r2`.

```powershell
python -m scripts.evaluate_semantic_state_bridge --smoke --output C:/CodexArtifacts/pretrained-reuse-r2/semantic-state-bridge-smoke-01
python -m scripts.evaluate_semantic_state_bridge --output C:/CodexArtifacts/pretrained-reuse-r2/semantic-state-bridge-01
python -m scripts.evaluate_semantic_state_bridge --output C:/CodexArtifacts/pretrained-reuse-r2/semantic-state-bridge-replay-01 --replay-reference C:/CodexArtifacts/pretrained-reuse-r2/semantic-state-bridge-01 --replay-sha256 e497e1139b6ee7660f620f1f1d93bef7c1d502e01327d1e778990123a47a56ea
```

Commands reject existing output directories. Use new directories for a later
reproduction; do not overwrite the ones above. The256images were SHA17-selected
before inference; source images/labels/cache/checkpoints and code hashes passed
before/after checks. Native full-run logits match the original saved training
logits exactly. The four-image smoke differs by at most2.84e-14, within the fixed
1e-10tolerance, with identical argmax. Text/image state-dict digests remain exact;
no trainable parameters or gradients. No raw-image encoder extraction was rerun.

Selection SHA256 `8a7ac8d6513c48d198d14c4be345d1081f4e1b9be7ff573fa23adf49c44a4b76`.
Effective config SHA256 `00ddadcfd175c4fa4ab8ec3f18a189950d865efae52f13f9d372b88e036680a4`.

**Accounting clarification:** the prospective protocol calls92,320forward pair
presentations "unique". That wording is inaccurate for distinct text inputs:
actual calls contain59,520distinct `(first,second)` string pairs and32,800repeat
presentations. Images have256distinct IDs/media hashes; image-question IDs number
1,024. Some different images share facts, and oracle/predicted inputs overlap.
The executed protocol is preserved; this correction does not change sampling,
inputs, any numerical result or forward counts.

| Condition | Actual pair presentations | Distinct string pairs within condition |
|---|---:|---:|
| Oracle | 40,960 | 40,320 |
| Predicted | 40,960 | 40,640 |
| No evidence | 160 | 160 |
| No question | 10,240 | 10,080 |
| Total, distinct counted across conditions | 92,320 | 59,520 |

No-evidence scores broadcast to256images; no-question scores broadcast to four
questions/image; donor score reuse adds no calls. These are controls, not extra
independent evidence. Independent CPU Torch cross-entropy agrees with saved NLL
within1e-12 for all five conditions. Accuracy numerators were also reconstructed
from saved raw scores/targets. No uncertainty claim treats1,536overlapping
question pairs as independent scenes.

## Actual resources and limits

| Process | Seconds | Peak CUDA allocation | Peak host working set |
|---|---:|---:|---:|
| Separate4image smoke | 9.9149 | 188,530,688B | 1,236,885,504B |
| Full fixed inference | 38.7620 | 190,711,808B | 1,280,765,952B |
| Full fresh replay | 39.1203 | 190,711,808B | 1,281,253,376B |

Measured process bodies total87.7972seconds. These are wall times including
loading/verification/CPU work, not GPU busy-time measurements. Cloud cost0.
Main run2026-10-15T03:38:06.477300Z to03:38:45.248688Z. RTX3080Laptop16GB,
Torch2.6.0+cu124, CUDA12.4, Transformers5.6.2, FP32SDPA/TF32off,4CPUthreads;
the effective config retains all queried package/driver versions. Full run1,443
forward batches,92,320pairs, maximum61tokens; no truncation. The4GiBVRAM guard
limits PyTorch allocation, with reserved memory separately recorded.

Actually loaded frozen models: text22,713,601 + image head61,480 =22,775,081
parameters. The cached visual features depend on the separate114,365,184parameter
VideoPrism. Including that dependency would be137,140,265 for this image/text
slice; this is not a measured all-modality resident product or package size.

## Decision and next action

Close this fixed diagnosis. Keep all existing checkpoints and tests. Do not
search question wording or change options to raise this score. Next define a
prospective conditional-semantics training/validation contract with different
state combinations, question wording and candidate descriptions, including
condition-change and no-condition controls. Learn how conditions select evidence
before assuming a larger visual encoder is required. Preserve native inputs and
time/crossmodal requirements in the full product plan; a four-field symbolic
router alone would not meet that objective.

Ruff passes; CPU suite1,575passed/1skipped; no-Torch CI simulation1,148passed/
109skipped; manifest/catalog validation passes. Read-only code review found no
Critical/Important issues. No live legacy process/config/corpus/checkpoint was
changed. Exact-commit GitHub CI is checked after publication.

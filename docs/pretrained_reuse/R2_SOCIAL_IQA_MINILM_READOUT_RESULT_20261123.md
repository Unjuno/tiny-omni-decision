# R2 Social IQa MiniLM frozen-readout result — 2026-11-23

## Result and disposition

A single, predeclared CPU experiment trained a 49,409-parameter candidate scorer
over frozen `sentence-transformers/all-MiniLM-L6-v2` features. On the
seed-17, context-grouped validation split, the selected epoch-8 checkpoint
scored **53.81% Accuracy, 0.9506 NLL, 0.5661 Brier, and 0.0154 ECE-15** on
6,659 three-choice questions. Uniform random choice is 33.33% Accuracy,
1.0986 NLL, and 0.6667 Brier. This establishes that a compact BERT-family
encoder plus a small scorer can use supplied choices for this English social
commonsense task. It is far short of the project's 95–98% aspiration and is
not evidence for text generation, multimodal fusion, or general Decision
quality.

Keep this as exploratory validation evidence only. The validation split selected
the epoch, and exact-context grouping cannot guarantee separation of paraphrased
ATOMIC source events. Do not tune further on these validation records. No
official publisher dev/test or project sealed audit was opened.

## Frozen inputs and implementation

- Dataset: pinned AI2 Social IQa train archive and seed-17 80/20 split; see
  [source screen](R2_SOCIAL_IQA_SOURCE_SCREEN_20261123.md).
- Train/validation rows: 26,751 / 6,659, with zero exact normalized-context,
  normalized-example, and normalized-question/options overlap.
- Train JSONL SHA-256: `b197d0f9069aa3317e6c05d41129e3c60dbe1d9cc3131db69df6d36cb4e585f9`.
- Validation JSONL SHA-256:
  `242ada6fad51fea726e8f222a0b52bda80179bf50daa6a96c047685f588adcfd`.
- Encoder revision: `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`; 22,713,216
  frozen parameters. Candidate scorer: 49,409 trainable parameters, giving
  22,762,625 parameters before quantization.
- Scoring input: `[context] [question + candidate option]`, three candidate
  logits per example. Maximum sequence length 256; fixed preflight found no
  truncation.
- Seed 17, AdamW, learning rate 0.002, batch 128 questions, up to 8 epochs,
  minimum validation NLL selection. Best checkpoint was epoch 8.
- Source code commit: `27f823715d921567cd89a52c5bfad9599e7ab6d7`;
  config SHA-256 `737e7c062f633c82a76022980bf03c5a282abc9e76772a56b486568f7dfdf5cc`.
- Runtime: Python 3.11, PyTorch 2.6.0+cu124, Transformers 5.6.2. Explicit CPU
  execution at two configured Torch threads; no GPU work was done. Feature
  extraction took 665.1 seconds; scorer training took 5.8 seconds.
- Saving and reloading the selected scorer reproduced validation metrics and
  probabilities exactly (maximum delta 0). Reversing the three candidate
  positions and restoring output order changed probabilities by at most
  `1.20e-7`.

Data, model weights, features, checkpoints, predictions, and the machine-readable
run receipt remain outside Git under
`C:\CodexArtifacts\pretrained-reuse-r2-local\`. Run-report SHA-256:
`870bd0c41e45c5b51978cc5ac8c02ca69b3f2535928cd7c218243a0ead2bb072`.

## Learning curve

| Epoch | Train CE | Validation Accuracy | Validation NLL | Brier | ECE-15 |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.9956 | 0.5240 | 0.9733 | 0.5805 | 0.0206 |
| 2 | 0.9699 | 0.5304 | 0.9680 | 0.5767 | 0.0250 |
| 3 | 0.9626 | 0.5351 | 0.9651 | 0.5753 | 0.0232 |
| 4 | 0.9594 | 0.5316 | 0.9636 | 0.5737 | 0.0207 |
| 5 | 0.9560 | 0.5340 | 0.9604 | 0.5717 | 0.0222 |
| 6 | 0.9513 | 0.5382 | 0.9568 | 0.5704 | 0.0225 |
| 7 | 0.9460 | 0.5400 | 0.9525 | 0.5676 | 0.0180 |
| 8 (selected) | 0.9425 | 0.5381 | 0.9506 | 0.5661 | 0.0154 |

Training CE continued to fall while validation NLL and Brier improved through
epoch 8. Validation Accuracy peaked at epoch 7 and dipped 0.20 percentage
points at epoch 8. This short curve does not show a strong loss/calibration
overfit signal, but checkpoint selection and validation exposure limit the
interpretation. Do not call this a clean generalization ceiling.

## Implication for BERT-family exploration

This one result is a positive but weak text-only proof of feasibility at a
small parameter count; it is not a reason to launch a broad BERT sweep. The
Japanese BERT-base results in
[the existing cross-task report](R2_TEXT_BERT_CROSS_TASK_RESULTS_20261123.md)
are separate tasks and are not directly comparable to this English Social IQa
result. Preserve BERT-base as a teacher/reference and MiniLM-L6 as a compact
candidate. Before another text training run, admit a fresh task or evaluate a
predeclared candidate once on genuinely untouched data. Neither result closes
the multimodal R2 gate or the project-wide parameter/package and quality goals.

## Artifacts

The complete report is at
`C:\CodexArtifacts\pretrained-reuse-r2-local\socialiqa-minilm-l6-frozen-readout-20261123-attempt-01\run-report.json`.
The selected scorer, final scorer, train/validation features, per-example
probabilities, and epoch history are in the same external directory and their
hashes are recorded in that report.

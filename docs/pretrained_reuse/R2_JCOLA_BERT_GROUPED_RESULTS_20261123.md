# JCoLA grouped BERT Decision results — 2026-11-23

## Outcome

The frozen Japanese BERT-base teacher and parameter-budgeted four-layer LINE
DistilBERT candidate both collapsed to the majority option on the source-held
out JCoLA evaluation. This is a negative result for this frozen-feature,
shared-option-scorer setup. It does not establish a general ceiling for BERT,
Japanese Decision models, or the project.

The publisher's in-domain validation file was not used because it shares all
ten source groups with training. Checkpoint selection used two source groups
held out from in-domain training. After both checkpoints were selected, each
model was evaluated once on the independent 685-example, 27-source official
out-of-domain development set. No publisher test or sealed audit data was
loaded.

## Run and reproducibility

- Source: [`osekilab/JCoLA`](https://github.com/osekilab/JCoLA), revision
  `736d9eef3af04bb17e4cf59adf01325973234ff9`, CC BY-SA 4.0.
- Exact model revisions: Tohoku `bert-base-japanese-v3` revision
  `65243d6e5629b969c77309f217bd7b1a79d43c7e`; LINE Japanese DistilBERT
  revision `c32d05753d0d4be0f60482291a9452ec7e6f7d68` (layers 0–3 retained).
  Their local weight-file SHA-256 values are recorded at the end of this report.
- Training repository HEAD: `e3e39aa11dbca72193a95bbd5a216b82a4aa1e71`.
  The experiment code was uncommitted during execution and is identified by
  its exact script SHA-256 below.
- Run directory: `C:\CodexArtifacts\pretrained-reuse-r2-local\jcola-bert-grouped-decision-20261123-attempt-01`.
- Local start observed from output-directory creation: 2026-11-23 07:04:17.975
  JST. Run manifest written at 07:05:34.650 JST; measured elapsed time 76.67 s.
- Effective config SHA-256:
  `0fb4f881bdf38ec3a03020c04903962909b8eb588c6babb2e50affaf588f82df`.
- Training script SHA-256:
  `426601b7c10179a0fe666814fe3da5605ea5959933fb90e695c8a0a0899a7550`.
- Hardware/runtime: NVIDIA GeForce RTX 3080 Laptop GPU, CUDA 12.4, Python
  3.11.9, PyTorch 2.6.0+cu124, Transformers 5.6.2, one CPU thread, FP32.
  No cloud use; compute cost $0.
- Model GPU allocations peaked at 0.460 GiB for BERT-base and 0.269 GiB for
  four-layer DistilBERT. These are per-process PyTorch peaks, not total-device
  peaks while Orbitune was resident.

The existing Orbitune idle supervisor detected the new GPU process, wrote its
own yield sentinel, and exited gracefully with code 0 at step 1,102,325 after
uploading the continuation checkpoint (SHA-256
`bfe88b70ffcb9e6a695462487f85401123f34615bb9a6a17ef8eb41d9225c301`). No
Orbitune process, checkpoint, data, or configuration was edited by this run.
The supervisor remained waiting for a fresh idle window afterward.

## Data accounting and integrity

| Use | Examples | Groups / sources | Labels 0 / 1 |
|---|---:|---:|---:|
| Readout training | 6,342 unique source-grouped train sentences | 8 source groups | 1,104 / 5,238 |
| Checkpoint selection | 577 unique source-grouped validation sentences | 2 held-out groups (`Kuroda_1965`, `Shibatani_1990`) | 46 / 531 |
| One-time OOD development evaluation | 685 | 27 unseen source groups | 183 / 502 |

Train and selection-validation had zero source and normalized-content overlap.
The OOD sources had zero overlap with all ten in-domain training sources and
zero normalized-content overlap. For each model, 13,838 train/selection option
pairs were encoded and 1,370 OOD option pairs were encoded; no pair was
truncated (maximum untruncated pair lengths were 77/82 tokens on
train/selection and 65/71 on OOD for BERT-base/four-layer DistilBERT).
The official in-domain validation file and official test were not loaded by
the run script.

The split assignments used by the run were independently recomputed and
verified by the training script. A preflight manifest field had hashed the
LF-only serialized assignment bytes before Windows converted the saved file
to CRLF. The run manifest correctly records the actual assignment-file hash
`1fb47f3fc1687308c79192da1ae0c2158bc2215a41abbb2c2e190062709cd43f`; the
preserved preflight manifest and a separate correction record explain the
newline-only hash discrepancy. The correction record SHA-256 is
`9cfb8075b75e5343f996432651fb726b5765c3264096f50d2775544733f364e2`.
No split assignment or sample membership changed.

## Fixed training setup

Both encoders were frozen. A shared `768 -> 128 tanh -> 1` scorer learned one
score per supplied option, with softmax over `[unacceptable, acceptable]`.
Seed 17, AdamW (learning rate 0.002, weight decay 0.0001), batch size 64,
maximum 20 epochs, patience 3, and checkpoint selection by source-holdout NLL
were fixed before training. The fit initialized a new readout; no checkpoint
was warm-started.

Each model completed five epochs / 500 readout optimizer updates. Validation
NLL selected epoch 2 / 200 updates for both. Thus the run-wide data exposure
and selected checkpoint exposure are different: 31,710 and 12,684 unique-row
visits per model respectively. The selected training CE and selection NLL
curves were:

| Model | Epoch 1 train CE / selection NLL | Epoch 2 train CE / selection NLL | Epoch 5 train CE / selection NLL | Selected epoch |
|---|---:|---:|---:|---:|
| Tohoku BERT-base v3 | 0.4787 / 0.3485 | 0.4643 / 0.2905 | 0.4515 / 0.2942 | 2 |
| LINE DistilBERT, first 4 of 6 layers | 0.4789 / 0.3542 | 0.4643 / 0.2819 | 0.4627 / 0.3312 | 2 |

Training CE decreased while selection NLL worsened after epoch 2, which is a
small-split overfit/generalization warning. The validation contains only two
independent source groups, and both models had 0.5 macro Accuracy there. No
training metrics are treated as a fixed-example generalization gap.

## One-time out-of-domain development metrics

| Model | Accuracy | Macro Accuracy | Unacceptable recall | Acceptable recall | NLL | Brier | ECE-15 | MCC | Mean confidence |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Tohoku BERT-base v3 | 0.7328 | 0.5000 | 0.0000 | 1.0000 | 0.6496 | 0.4293 | 0.1445 | 0.0000 | 0.8773 |
| LINE DistilBERT 4-layer | 0.7328 | 0.5000 | 0.0000 | 1.0000 | 0.6705 | 0.4384 | 0.1570 | 0.0000 | 0.8898 |

The majority baseline is also 502/685 = 0.7328 Accuracy. Both models predicted
“acceptable” for every example; neither correctly predicted any of the 183
unacceptable examples. Across the 24 source groups containing at least one
unacceptable item, both had zero unacceptable true positives. Macro Accuracy
averaged equally across all 27 source groups was 0.7123 for each model.

A paired source-cluster bootstrap on the saved predictions (27 source groups,
5,000 resamples, seed 20261123, percentile 95% interval) found four-layer
DistilBERT minus BERT-base Accuracy difference 0.0000 [0.0000, 0.0000], NLL
difference +0.0209 [+0.0077, +0.0347], and Brier difference +0.0092
[+0.0037, +0.0150]. The smaller candidate has the same collapsed decisions
and worse probability quality in this comparison.

## Interpretation and next work

The four-layer candidate fits the current four-tower arithmetic at 195,994,241
pre-quantization parameters including this small readout, but leaves only
4,005,759 parameters before product-specific heads. Its text-only encoder plus
readout is 54,010,625 parameters. The BERT-base text-only encoder plus readout
is 110,715,137 parameters and does not fit the current four-tower stack.

This BERT-family branch is **not a quality pass**. It demonstrates that a
frozen CLS pair scorer with this label distribution does not learn the
unacceptable class under source-held-out selection, even though it produces
normalized option probabilities. Class imbalance and source/domain shift are
visible contributors, but this single run does not isolate their causal role
from the frozen representation or scorer design. Do not adjust class weights,
thresholds, or prompts against this now-exposed OOD development set. The next
quality experiment requires a different source set or a newly frozen group
holdout and a preregistered training/scoring change.

No quantization, multimodal fusion, product promotion, or 95–98% claim follows
from this text-only result. The overall R0–R8 objective remains open.

## External artifacts and checksums

| Artifact | SHA-256 |
|---|---|
| Run manifest | `3771fe87201048e5829b358a7f52fff3fdf66c0f86f413ad1dca27abcfe12cd9` |
| BERT-base run report | `bd555b6cfc5bd051c59a48af81617a8da35b4cd9d288d8a444844a28897d546b` |
| BERT-base selected readout | `72a4748ea0323dcaa8ad4f6f78f23f325fe5155cd837997dfa126692dcb2964e` |
| BERT-base OOD predictions | `b2ac3f0b6bfa699fba5f79b5c19fa540332545d88b178d319af9440442920d9f` |
| Four-layer DistilBERT run report | `7198bfc15b210934d9ad98d8e142d67227d929eb234754078b5147e8a17a30f7` |
| Four-layer DistilBERT selected readout | `ccaa97c28492e28293da8a627c45a02b49822118c4e6e4e32673b7d30d07cfe6` |
| Four-layer DistilBERT OOD predictions | `2d50c08b79acc760e6fb315de0300e5c1f8608bb52736e53190fbfbf3cdb94d7` |
| Saved-prediction analysis | `2ca0af27bb5328dac346643b1762950ff0da605080914c7348625700583692ef` |

Model weight SHA-256 values were BERT-base
`e172862e0674054d65e0ba40d67df2a4687982f589db44aa27091c386e5450a4` and LINE
DistilBERT `9d8b678bd9ce14363d5e6e3dc53c7dd7c76f8ad11394ccffd200f15e9f6819eb`.
They match the previously pinned local checkpoints; no model weights or raw
data were added to Git.

Focused CPU tests: 6 passed; Ruff passed. Full CI is pending after this report
and scripts are committed.

## Smoke-attempt log

The fixed experiment was reached only after retaining these setup attempts:

| Attempt | Outcome |
|---|---|
| 01 | Stopped before fitting: system Python lacked `fugashi`. |
| 02 | Stopped before fitting: the alternate system environment lacked `sentencepiece`. |
| 03 | Loaded both models and completed a tiny fit, then failed serializing set-valued loading diagnostics into the smoke manifest. |
| 04 | CPU plumbing smoke passed after manifest serialization was fixed. |
| 05 | Final CPU smoke passed after token-ID bounds and source-group contracts were added; it did not load or score OOD records. |

These were code/environment checks, not extra quality evaluations. The full
comparison is the single OOD development exposure described above.

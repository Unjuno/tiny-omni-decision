# QASC same-state, new-question Decision diagnostic — 2026-11-05

## Claim and disposition

This is a train-only, text-only diagnostic of whether a frozen text-state representation helps answer a different question about a fact state already represented in training. It is not a clean project-level holdout, broad text capability, multimodal fusion, temporal memory, or product-quality result. Do not repeat the fit or use it to select a product architecture.

The result supports a narrow mechanism claim: on the predeclared held-question slice, supplying the correct evidence state and criterion performed better than removing the criterion or replacing/zeroing the state. The result does not isolate a reusable latent state from lexical answer leakage: the target string occurs in the evidence for 542/683 held rows (79.36%). The target-absent stratum has only 141 rows and remains an internal development diagnostic.

## Frozen run and integrity

- Source: `allenai/qasc` prepared train data at revision `a34ba204eb9a33b919c10cc08f4f1c8dae5ec070`, CC-BY-4.0; official TRAIN only.
- Source corpus SHA-256: `6ad3536ebba70a35f87bf078d0d2837df09fda652201a2b4a6fd7d7b01ed7f0b`.
- Split: 7,451 train rows and 683 held-question rows across 666 fact-state groups. State overlap is intentional: every held state is represented in training, while normalized question overlap and normalized question-plus-option overlap are both zero. This tests a new question on a known text state, not transfer to unseen states.
- Exposure: every QASC TRAIN row had already appeared in the preceding five-fold train-only experiment universe. The official validation shard, official test shard, and sealed audit were not read. Therefore these held rows are not independent project-level evaluation data.
- Model: frozen `sentence-transformers/paraphrase-MiniLM-L3-v2`, revision `4ca70771034acceecb2e72475f72050fcdde4ddc`, 17,389,824 frozen encoder parameters; `FusionDecisionScorer` readout with 164,610 trainable parameters. One seed (17), fixed 8-epoch endpoint, 936 optimizer updates, AdamW at 0.001, weight decay 0.0001, CPU, 4 Torch threads. No held-question evaluation occurred during fitting.
- Run window: `2026-11-04T16:04:13Z` to `2026-11-04T16:06:40Z` (2026-11-05 local time); training 8.05 s; final evaluation 0.48 s; elapsed through final evaluation 147.75 s.
- Source code commit: `0858f851ea4e7781880c3f46c5634e4e1a23d251`. Encoder weight SHA-256: `cf1e4e2d420c664973037c3c73125d7a8fc69952495093ef8f50596f8943a433`.
- Effective-config SHA-256: `a6723b896101529122dec440a8490b9da45cad12ac68ec3dd03e25133104552b`; experiment-config SHA-256: `c9310fd44b88cefbe843845f10fdbdd4ee31dcffe8070231143785ee8549b88c`; runner SHA-256: `711c276178171416f5ca689b94fdc15156cc7cb292c13fd8bed56410103cc61f`; split-helper SHA-256: `1f0b178f69ce47b617b08f1f2f8ad5ca35222b4f0f26d4f2509ab5d27b7d2554`.
- Checkpoint SHA-256: `6407ad1c2e24ff099332f7c9c347baec7f4e6daee721b82dcf9ca56f18db0e20`; prediction SHA-256: `7138c49afb424e615aa1c95b08cc0e733368010f4ed49ab18ffaad49a6de9650`.
- Run report SHA-256: `a0ee840d284fdcbfa09877b4e32bcb0f3ce2636a9abbd873fc04f1283a271521`; target-absent sensitivity receipt SHA-256: `8fe32f0e07d8633446b32dfb405082870d3b4191e447f475a93c5a9d5087092b`.
- External immutable run directory: `C:/CodexArtifacts/pretrained-reuse-r2/qasc-same-state-new-criterion-20261105-attempt-01/`. Raw dataset, feature matrix, predictions, and weights remain outside Git.

## Results

| Held-question condition | n | Accuracy | NLL | Brier | ECE-15 |
|---|---:|---:|---:|---:|---:|
| Correct state + criterion | 683 | 78.62% | 0.7065 | 0.3205 | 0.0653 |
| Correct state, criterion absent | 683 | 70.13% | 1.1174 | 0.4605 | 0.1378 |
| Zero state content + criterion | 683 | 15.37% | 4.7872 | 1.2745 | 0.5437 |
| Deranged different state + criterion | 683 | 15.96% | 5.7429 | 1.3608 | 0.5986 |
| Uniform eight-choice control | 683 | 12.74% | 2.0794 | 0.8750 | 0.0024 |

For correct state + criterion versus correct state with criterion absent, the paired 5,000-replicate bootstrap over 666 held state groups (seed 1718) gives an Accuracy difference of +8.49 percentage points (95% percentile interval +5.34 to +11.62), NLL difference −0.4096 (−0.5181 to −0.3074), and Brier difference −0.1397 (−0.1771 to −0.1041). These intervals describe this development slice only and do not account for the prior use of all source TRAIN rows in related experiments.

In the post-hoc target-not-in-evidence slice (141 rows / 141 groups), correct state + criterion scored 72.34% Accuracy, 0.8770 NLL, and 0.3943 Brier. Correct state with criterion absent scored 63.12%, 1.3518, and 0.5337. The paired group-bootstrap differences are +9.12 percentage points Accuracy (95% interval +2.13 to +16.31), −0.4762 NLL (−0.7032 to −0.2665), and −0.1387 Brier (−0.2176 to −0.0648). This small post-hoc stratum is a sensitivity check, not a new selection set.

Checkpointing used one fixed epoch-8 readout, whose save/reload probabilities were verified exactly; there was no checkpoint-selection sweep.

## Interpretation for R5 and next work

The paired controls indicate that the known text evidence state contributes information for a different held question, beyond the criterion-free same-state condition. However, the state was text, already present during training, 79.36% of targets were literally present in the evidence, and the source train rows had prior project-level exposure. This cannot establish persistent temporal memory, cross-modal state, unseen-state generalization, or useful product accuracy.

This result does not close R5's persistence or temporal-recall gate and does not justify adding an LSTM, state predictor, or shared multimodal architecture. Preserve it as a bounded text mechanism result; any next state-reuse claim needs a newly admitted task with target-absent evidence, independent state groups, delayed questions, and a frozen evaluation split.

No new model training, raw dataset/media access, prediction generation, or sealed-audit evaluation was performed while preparing this repository record. The report was reconstructed from existing metadata, metrics, and prediction receipts; the completed run artifacts were not modified.

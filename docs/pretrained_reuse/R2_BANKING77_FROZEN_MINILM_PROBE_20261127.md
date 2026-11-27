# BANKING77 frozen MiniLM probe — 2026-11-27

## Decision and scope

This is one CPU-only, frozen-feature text probe on a deterministic development
split derived from the publisher's train partition. It asks whether a compact
BERT-family representation can separate BANKING77's 77 English banking intents
with a small readout. It does **not** establish arbitrary-option Decision
quality, general language quality, multimodal quality, or the project's 95–98%
product target. At the time of this run, the publisher test and project sealed
audit were unopened; the publisher test was subsequently evaluated once under
the separate [one-shot report](R2_BANKING77_MINILM_OFFICIAL_TEST_20261127.md).

The source is CC BY 4.0 according to the pinned dataset card and repository
license; attribution is required. The card reports expert-generated categories,
but leaves source-producer details unresolved and provides no stable
user/session identifiers. See the [train-only data screen](R2_BANKING77_TRAIN_DATA_SCREEN_20261127.md)
and the [pinned dataset card](https://huggingface.co/datasets/PolyAI/banking77/tree/1846984189f2bc0ac51c6fc3a600fa9e1714e0c6).

## Data review and frozen split

Only the pinned `train.csv` was read (10,003 records, 77 labels). The 57
cross-intent lexical candidates were read in context. Most express meaningful
fine-grained modifiers such as physical versus virtual card, card payment
versus direct debit, or transfer versus cash/card top-up. Several generic
versus specific phrasings remain ambiguous. One pair is the same after
punctuation normalization but has labels `pending_top_up` and `top_up_reverted`.
No text was dropped and no label was changed. The dataset's class-overlap and
provenance limits remain part of the result.

To reduce near-duplicate leakage, connected components of all 786 lexical
candidate pairs (plus exact normalized duplicates) were assigned wholly to one
split. The greedy class-balanced allocation used seed 17 and a 20% validation
target. The realized split is 7,903 train / 2,100 validation (21.0%), with all
77 intents represented on both sides and **zero candidate-component overlap**.
Validation has 7–38 examples per intent; train has 28–151. This is only a
lexical grouping proxy: the corpus has no user/session IDs, so independent
customer or event generalization is unverified.

The ID-only split manifest is
[`manifests/data/banking77_train_grouped_validation_v1.json`](../../manifests/data/banking77_train_grouped_validation_v1.json),
SHA-256 `b8187c8f57c2f567803526938c3fde89c8b4d120637d507ae405bedffba9b29b`.
Its validation row-ID hash is
`4818172cc8a620854a95e43780ee3eba30c79c1a09b6803ad181c3897b066316`.

## Model and procedure

- Checkpoint: `sentence-transformers/paraphrase-MiniLM-L3-v2`, revision
  `4ca70771034acceecb2e72475f72050fcdde4ddc`; the pinned card declares Apache
  2.0. The [official model card](https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L3-v2/tree/4ca70771034acceecb2e72475f72050fcdde4ddc)
  describes a 384-dimensional sentence embedding model.
- Pinned local `model.safetensors` SHA-256:
  `cf1e4e2d420c664973037c3c73125d7a8fc69952495093ef8f50596f8943a433`.
- Runtime count: 17,389,824 frozen encoder parameters plus 29,645 trainable
  linear-readout parameters = **17,419,469 total**. The encoder remained frozen.
- Input: query text, max length 128, attention-mask mean pooling. The head emits
  probabilities over the fixed 77 source intents; it is not a text generator or
  an open-vocabulary option scorer.
- Seed 17; AdamW, learning rate 0.01, weight decay 0.01, batch size 64; up to
  100 epochs. Best selected by validation macro NLL with `min_delta=1e-4` and
  patience 10. Best epoch 10; the run continued to epoch 20 before stopping.
- CPU only on an Intel i7-12700H; no GPU or cloud compute. Frozen feature
  extraction took 17.40 seconds; head fitting took 3.95 seconds. The separate
  save/reload verification recomputed features on CPU and reproduced every
  validation probability exactly (maximum absolute delta 0.0).
- Runtime environment: Python 3.11.9, PyTorch 2.6.0+cu124 (CPU execution),
  Transformers 5.6.2, safetensors 0.8.0, Windows 10.

Run implementation: [`run_banking77_minilm_probe.py`](../../scripts/run_banking77_minilm_probe.py).
The execution-time copy and full environment/config/output hashes are preserved
outside Git at
`C:\CodexArtifacts\pretrained-reuse-r2-local\banking77-frozen-minilm-l3-seed17-attempt-01\`.
Execution receipt SHA-256:
`bd66a146b9d16afd76eebd43ac92cff4f59fe54c5cee981ada56f144b3eba539`.

## Results

| Metric | Majority baseline | Frozen MiniLM + linear readout |
|---|---:|---:|
| Validation examples | 2,100 | 2,100 |
| Sample Accuracy | 1.71% | 92.24% |
| Macro Accuracy | 1.30% | 92.23% |
| Sample NLL | 4.3487 | 0.3016 |
| Macro NLL | 4.3529 | 0.3172 |
| Multiclass Brier | 0.9873 | 0.1177 |
| Macro multiclass Brier | 0.9875 | 0.1179 |
| Top-label ECE (15 bins) | 0.0174 | 0.0157 |
| Macro classwise ECE | 0.9870 | 0.1174 |
| Mean confidence | 3.45% | 91.97% |

The reloaded readout reaches 98.28% sample / 98.40% macro train Accuracy,
versus 92.24% / 92.23% on validation: a 6.04-point sample-accuracy gap. Train
NLL is 0.1014 versus validation NLL 0.3016. Validation macro NLL selected
epoch 10; no later checkpoint improved it. This is evidence of overfitting in
the readout fit, even though validation accuracy remained high.

Largest validation confusions by count include `declined_card_payment` →
`reverted_card_payment` (6), `balance_not_updated_after_bank_transfer` →
`transfer_not_received_by_recipient` (5), `card_payment_wrong_exchange_rate`
→ `wrong_exchange_rate_for_cash_withdrawal` (4), and `pending_top_up` →
`top_up_failed` (4). These align with fine-grained, sometimes overlapping
labels; they do not by themselves prove bad annotations. The full per-intent
metrics and per-example probabilities are retained in the external run output.

Readout SHA-256:
`2b5d186aa1be350651c109d4dae84fc03764a28446965c476b94349b5df34697`;
118,724 bytes. Prediction file SHA-256:
`439169768ca103312ccbcea2e47054839a6c83a7fe08a037382492c95869d8a8`.
The encoder weights are 69,569,488 bytes. This is a small text-only candidate,
not a complete multimodal product package.

## Interpretation and next gate

This result supports a **targeted BERT-family text path**: a 17.4M-parameter
frozen MiniLM representation plus a small readout performed strongly on a
source-grouped 77-intent development task. It does not justify a broad BERT
sweep. The project already has mixed BERT-family findings on other tasks; this
new narrow result complements them without overriding them.

The 95–98% aspiration was not reached, and this validation is now exposed by
model selection. Do not tune further on it or promote this checkpoint. The
publisher test has since been scored once, with no selection or tuning; its
result and now-exposed status are documented separately. The project's sealed
audit remains closed. Before another model comparison, resolve the task/data
question with an independently reviewed source-grouped task or an independently
adjudicated label subset. Preserve the known conflicting pair, report ambiguity
explicitly, and do not repair labels by model-driven filtering. No bootstrap
confidence interval or second seed was run.

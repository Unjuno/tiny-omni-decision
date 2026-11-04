# QASC same-state / new-criterion diagnostic protocol (2026-11-05)

## Estimand

Measure whether one small text Decision scorer can use a repeated pair of
publisher-provided evidence facts to answer a different, previously untrained
question and its supplied eight choices. The exact normalized fact-pair state
appears in readout training under other questions; the normalized held question
and its question-plus-option-set must not appear in training. This tests a
narrow form of same-observation/many-question reuse from the ROADMAP. It is not
unseen-state generalization, temporal world modeling, persistent-cache runtime
reuse, or multimodal fusion.

## Data and frozen split

Use only 8,134 CC-BY-4.0 official QASC TRAIN rows from
`allenai/qasc@a34ba204eb9a33b919c10cc08f4f1c8dae5ec070`, prepared corpus SHA-256
`6ad3536ebba70a35f87bf078d0d2837df09fda652201a2b4a6fd7d7b01ed7f0b`.
For normalized evidence-state groups with at least two distinct questions,
select one question that is globally unique to that state. Normalize states/questions with the repository NFKC-aware `_normalized`
function; iterate states and candidate questions in sorted order, using `random.Random(17)` to select one
eligible criterion per state, then hold all rows for the selected question.
The fixed expected split is 7,451 training rows and 683 held rows across 666
states. Each held state must remain in training through another question;
normalized question and normalized question-plus-option-set overlap must be
zero. Frozen ordered train/held ID hashes are respectively
`6d7408b3cf5bb58a6dabe753ed88d90a744645564af60b87df1d144c4bb460cf` and
`d44f521df10ddee02e026613438c30453eb4c89ccca5683a9e687062ab2f1918`; fail if
they differ. The state overlap is intentional and is the point of this estimand.

Before fitting, save the effective config, source/corpus hashes, ordered train
and held ID hashes, exact split plan, group identities, and target-in-evidence
counts. Fail closed if any expected count/hash/invariant differs. The official
QASC validation, test, and sealed audit shards are forbidden inputs.

## Frozen model and fit

Use the existing frozen MiniLM-L3 revision
`4ca70771034acceecb2e72475f72050fcdde4ddc` (weight SHA-256
`cf1e4e2d420c664973037c3c73125d7a8fc69952495093ef8f50596f8943a433`). Encode
state, question criterion, and each supplied option with the existing
256-token masked-mean contract. Freeze the encoder and use a fresh
`FusionDecisionScorer` with 164,610 trainable parameters. Seed 17, AdamW
`lr=1e-3`, weight decay `1e-4`, batch 64, exactly 8 epochs, CPU. Do not inspect
the held rows during training and do not select a checkpoint from them. Save
only the fixed epoch-8 endpoint and rolling training loss.

## Fixed same-ID controls and reporting

Evaluate the held rows once after epoch 8 under identical IDs/options/targets:

1. correct evidence state plus held criterion;
2. correct state plus criterion absent;
3. zero state-content vector with the state slot still present (a content ablation;
   the scorer's API rejects an all-missing input);
4. criterion plus a deterministic Sattolo derangement of a different held
   state feature for every held state.

Report Accuracy, NLL, multiclass Brier, ECE-15, mean confidence, target-in-state
strata, option-position controls, and per-row probabilities. Compute paired
5,000-replicate percentile bootstrap intervals by held state for differences
between correct state and each control. Run option-order invariance and exact
save/reload prediction checks on fixed inputs. Do not use any held result to
change data, code, training, or checkpoint selection.

Call state use supported only if correct-state Accuracy is higher and NLL is
lower than both zero-state and shuffled-state controls, and paired 95% cluster
intervals exclude zero in the favorable direction for both Accuracy and NLL.
Otherwise report mixed/no support. The selected held target string occurs in its supplied evidence in 542/683 rows
(79.36%); in the parent corpus the rate is 75.82%. so report target-in-evidence
and other strata and retain lexical shortcut as a competing explanation.

## Exposure and claim boundary

Every QASC TRAIN row appeared in the preceding five-fold experiment universe,
so the held questions are only new under this preregistered within-state split;
they are not project-level unseen examples or independent quality evidence.
This is a text-only development diagnostic. It cannot establish persistent
feature-cache behavior, natural audio/image/video/sensor quality, a multimodal
world representation, or product accuracy. No official validation/test or
sealed audit is opened. No additional seed, architecture, or budget sweep is
planned.

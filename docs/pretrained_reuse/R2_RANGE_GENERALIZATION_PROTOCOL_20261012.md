# Fixed selected-weight range and wording comparison

Purpose: test whether expanded-range training fit transfers to unused states,
without treating training Accuracy as general reasoning. This is a development
diagnostic chosen after observed failures, not a new blind product audit.
User-authorized autonomous local work; no new training, seed search or paid compute.

## Frozen contenders

- Parent narrow encoder-coverage selected2048 SHA256
  `4024788b1a9d17735612a98da6acb1a65af5d82a9d440b4fddb4ab62efb9d318`.
- Range warm-start selected1408 SHA256
  `f287bb0138fd0f264331cd2fc983567bc68323a85a25cda467387fffae7bc93b`.

Use identical pinned MiniLM base/tokenizer and question-first FP32 evaluation.
Neither final2048 nor alternative checkpoints enter this comparison. Both models
have22,713,601parameters. Extra inherited training differs; this is a practical
candidate comparison, not equal-compute causal identification of initialization.

## Inputs fixed before running either model

Seed97,256new latent states, eight questions per state. Same four items, option
counts, option positions, underlying answers and row/key order across arms.
Five arms,2,048questions each:

1. Control: original generator values1..199 and original templates.
2. Interpolation: all cost/risk/time values and risk thresholds become7*x+19.
   Values26..1412 lie within the expanded training envelope, but that affine
   transform was not trained. This is not a claim that individual values are unseen.
3. Extrapolation: values and thresholds become13*x+29, values42..2616.
   Report how many individual values exceed the training maximum1592; this arm
   mixes within-envelope and outside-envelope values, not pure extrapolation.
4. Wording: original numbers, fixed new templates below.
5. Combined:13*x+29 plus those templates. Interactions are not separately identified.

Templates:

- Extremum/time: `Which item has the smallest/largest <field>?`
- Count: `Return the number of items with risk less than or equal to <threshold>.`
- Conditional: `Exclude items whose risk exceeds <threshold>. Of the remaining
  items, which has the smallest cost? Answer none if no items remain.`

Reject latent-state group overlap with original train/validation and previous
seed43/seed59 transfer states, plus duplicate IDs/normalized content against the
prior generated corpora. Do not remove hard questions or regenerate a favorable
seed. Independently verify transformed labels, preserve target positions and
write input hashes and all IDs before inference. An integrity failure blocks the
evaluation until the implementation is corrected and recorded.

## Evaluation and interpretation

Run both fixed models on all five arms once. No truncation: fail if any pair
exceeds512tokens. Save IDs/options/targets/probabilities and Accuracy/NLL/Brier/ECE
overall and by task, input lengths, wall time and peak VRAM. Compare candidate
minus parent with paired state bootstrap10,000replicates, seed20261012, percentile
95% intervals; describe selection/multiple-comparison limitations explicitly.
Report each arm rather than choosing the best rendering. No threshold fitting.

Evidence of numeric transfer requires consistent candidate improvements on the
numeric arms; report mixed outcomes and control/wording regressions separately.
Neither training fit nor one favorable arm establishes generalization. These
synthetic text tasks cannot satisfy image/audio/video or the95% product goal.
No old diagnostic inputs, sealed audit or model promotion. New outputs only.

Implementation must include CPU tests for deterministic generation, oracle and
target-position preservation, overlap rejection, fixed checkpoint hashes and
probability accounting before local inference. Existing artifacts remain intact.

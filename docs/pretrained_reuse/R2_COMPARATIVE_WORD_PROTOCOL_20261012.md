# Fixed comparative-word factorial diagnosis

Known development failures motivate this diagnosis; not blind validation and not
a smaller replacement for the full benchmarks. Keep their weak results intact.
No training, calibration, checkpoint selection or product claim.

Same fixed numeric-only/lexical-range selected1408weights as lexical transfer.
Seed137,128new states, four cost/risk min/max questions/state. Count, conditional
selection and scalar time are outside this focused mechanism diagnosis; no
modality-wideAccuracy or95%claim may be made from it.

Eight paired conditions,512questions each: numeric values unchanged or7*x+19;
comparison word minimum/maximum or smallest/largest; sentence structure
`Select the item with the <word> <field> value.` or
`Which item has the <word> <field>?`. These templates are intentionally related
to the observed failures, not novel-template claims. All states are new;
answers, candidate order and question identities are paired across conditions.
Reject overlap with all prior generated states/content; independent oracle check.

Save input hash before inference. Same FP32/local processor, native option scores,
no truncation, all probabilities and per-condition metrics. Primary diagnosis:
within each numeric range and sentence structure, smallest/largest minus
minimum/maximum Accuracy/NLL/Brier/ECE. Secondary: structure effect at fixed word
and numeric range; compare both models on identical inputs. Pair by128states,
10,000bootstrap resamples,seed20261012,percentile95% intervals. Multiple diagnostic
comparisons and prior selection are not corrected; report limitations.

No favorable wording selection or prompt change after viewing outputs. A word
effect supports vocabulary sensitivity in this setup, not a unique causal
explanation of all decision failures. Preserve all previous results and weights.

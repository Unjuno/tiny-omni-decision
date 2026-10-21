# Soft-replay paired uncertainty

Saved fixed-endpoint predictions only; no fitting or selection. State-paired
bootstrap,10,000 replicates,seed17,percentile95%,unadjusted intervals.732 original
states/2540 questions;128 context states/768 questions per condition. Every ID,
option order and target matched before analysis. Positive NLL/Brier differences
are worse; positive Accuracy differences are better.

| Comparison: soft minus baseline | Accuracy delta (95% CI) | NLL delta (95% CI) |
|---|---|---|
| Original vs hard replay | +.01457 [.00355,.02591] | -.26463 [-.30890,-.22120] |
| Original vs parent | +.00354 [-.01289,.02018] | +.02419 [.00681,.04139] |
| Necessary context vs hard replay | +.01563 [.00387,.02865] | -.02691 [-.05019,-.00433] |

Original Brier versus hard improves -.06532 [-.07860,-.05236]. Versus parent,
Brier difference+.00829 [-.00150,.01786] is unresolved. Plain and irrelevant
context contrasts versus hard have intervals spanning zero for all three metrics.
Full contrasts and input hashes are in the adjacent JSON.

The soft-target intervention improved the hard-replay comparator within this
single-seed development experiment. It does not establish parent dominance:
original-task NLL remains worse. These repeated development analyses are neither
independent blind audits nor multiplicity-adjusted confirmation. Do not select a
checkpoint or tune temperature/mixture weights from these intervals.

Close this fixed comparison with both outcomes retained. Before any further
training, the existing frozen vocabulary-transfer/range diagnostics can test
whether the context improvement transfers; their observed status must remain
explicit. No repeat of the same fit or additional seed is justified solely to
make the remaining gap disappear. Multimodal/fusion gates remain unmet.

Reproduce: `python -m scripts.compare_soft_replay`.

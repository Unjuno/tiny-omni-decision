# Predeclared full-coverage conditional repair experiment

Decision after the completed512-update pilot: do not infer a capacity ceiling
from25% curriculum coverage. User authorized autonomous local experiments.
Run one2048-update experiment from the same preserved adapted epoch3 weights
and new AdamW, not from the pilot checkpoint. Same seed17, constant2e-5 LR,
final block/classifier only, dropout disabled, equal4+4 question batches,
corpus hashes, validation IDs/order, every128-step evaluation and mean-NLL
selector including step0. No architecture/data/prompt/precision changes.

This covers8,192 distinct curriculum questions and8,192 distinct original
questions,16,384 total, without question repetition. The schedule's first512
updates equal the pilot exactly; a CPU test verifies this and full coverage.
Compare the initial and prefix metrics to the pilot; report any numerical
mismatch without claiming exact training resume. Save every scheduled evaluation
and predictions plus separate best/final deltas. No performance-based early stop;
stop on numeric, integrity or resource failures. No seeds/budget beyond2048.

Predeclared interpretation: compare512 vs2048 within this run, including1024/
1536intermediate points. Small or mixed changes are UNCERTAIN; no gain means
no support for useful budget scaling here, not a universal model limit. Preserve
all four task groups and original-corpus retention. Near-uniform metrics cannot
qualify as useful reasoning, regardless of selector improvement. This remains
an observed synthetic development experiment, not the product95% evaluation.

Output C:/CodexArtifacts/pretrained-reuse-r2/conditional-repair-full-coverage-01.
CPU/smoke checks precede execution. Runtime estimate from pilot: roughly15minutes
including17scheduled evaluations and best reload, local GPU only. No cloud cost.
Current disk free4.4GB; expected new outputs well below100MB.

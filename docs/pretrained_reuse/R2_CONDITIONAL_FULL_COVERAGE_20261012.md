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

## Completed measurement

Execution **PASS**, quality **not demonstrated**. Source commit 6c5b723;
command `python -m scripts.train_conditional_repair --root C:/CodexArtifacts/pretrained-reuse-r2 --model C:/CodexArtifacts/pretrained-reuse-r1/crossencoder-minilm-l6-audit-01/model --output C:/CodexArtifacts/pretrained-reuse-r2/conditional-repair-full-coverage-01 --steps 2048`.
All 17 scheduled evaluations and per-question probabilities saved. No performance
stop; best selection remained step 768, separately from final step 2048.
Exact best-logit reload and unchanged frozen-parameter hash passed.

Run-wide consumption: 16,384 unique questions, 8,192 per corpus, no repeated
question; all generated training questions covered. Selected step 768 consumed
6,144 unique questions, 3,072 per corpus. Do not confuse those amounts.
The independent validation IDs/order and first 4,096 consumed IDs exactly match
the pilot. Maximum prefix mean-NLL difference: 2.98e-8; not a strict resume.
Only the existing 1,774,849 final-block/classifier parameters were trainable;
22,713,601 total model parameters unchanged. New AdamW from preserved epoch3 delta.

### Fixed validation curve

| Step | Original Accuracy | Original NLL | Curriculum Accuracy | Curriculum NLL | Selector |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.58110 | 0.93779 | 0.23389 | 1.45684 | 1.19731 |
| 512 | 0.57835 | 0.93659 | 0.24365 | 1.44972 | 1.19316 |
| 768 | 0.57953 | 0.93184 | 0.23291 | 1.44684 | 1.18934 |
| 1024 | 0.57244 | 0.93837 | 0.25391 | 1.44696 | 1.19266 |
| 1536 | 0.57047 | 0.94778 | 0.23730 | 1.44508 | 1.19643 |
| 1920 | 0.56969 | 0.94606 | 0.23291 | 1.44415 | 1.19511 |
| 2048 | 0.57283 | 0.93642 | 0.23145 | 1.44437 | 1.19040 |

| Checkpoint/corpus | Accuracy | NLL | Brier | ECE (15 bins) |
| --- | ---: | ---: | ---: | ---: |
| selected768/original | 0.57953 | 0.93184 | 0.53136 | 0.02998 |
| selected768/curriculum | 0.23291 | 1.44684 | 0.76511 | 0.03704 |
| final2048/original | 0.57283 | 0.93642 | 0.53222 | 0.03912 |
| final2048/curriculum | 0.23145 | 1.44437 | 0.76400 | 0.03520 |

### Interpretation and limits

Predeclared result **UNCERTAIN / no useful accuracy gain demonstrated within this
budget**: from 512 to 2048 mean NLL improves slightly, but both corpus accuracies
fall. Curriculum NLL/Brier remain worse than a uniform distribution (NLL 1.44208,
Brier 0.7625). Expected uniform random Accuracy is 23.75%. Selected checkpoint
curriculum Accuracy is 23.29%; final is 23.14%. This is not useful conditional
reasoning, despite a numerically better selector. Best step 768 followed by
fluctuating original metrics suggests no sustained retention gain. Rolling
training CE is over changing samples and is not a fixed-set generalization gap.

Paired state-cluster bootstrap, 256 states with all eight questions kept together,
10,000 replicates, seed 20261012, percentile 95% interval: final-minus-512 Accuracy
-1.22 percentage points, interval [-2.98,+0.54]. This observed development set also
serves checkpoint selection; the interval does not make it a blind evaluation.
No capacity ceiling, impossibility of future improvement, or modality quality
claim follows from this narrow experiment.

Next direction: stop merely extending this last-block run. Test whether allowing
more of the existing small encoder to adapt changes conditional learning, keeping
initialization, sample order, data, optimizer and fixed evaluation identical.
Any such run needs a fresh output and separate frozen-parameter checks. Keep
new model parameter count unchanged and retain original-task trade-offs. This
is a proposed next comparison, not an executed or scheduled result. It must not
replace broader text/image/audio/video task-generalization gates.

### Resources and preservation

Local RTX 3080 Laptop 16GB, torch 2.6.0+cu124, Transformers 5.6.2, FP32. Timed
training/evaluations/best reload: 884.70 seconds; loading/tokenization precede this
timer. Peak PyTorch allocated VRAM: 802,154,496 bytes (not whole-device memory).
No cloud cost. Start UTC in immutable config; every evaluation elapsed time,
resource counts and task metrics are in the adjacent JSON and external artifacts.
Best and final weight files retained separately with SHA-256. No optimizer/RNG
resume state is saved: weights-only artifacts do not support exact resume.
Legacy model/data/artifacts unchanged; sealed audit unopened.

Ruff passed; CPU tests 324 passed, 1 skipped; model/dataset/catalog validation and
license-manifest audit passed. Learning-curve values above and all 17 points in
JSON are authoritative. Optional PNG rendering was not produced because the local
environment lacks matplotlib; no dependency upgrade was made for presentation.

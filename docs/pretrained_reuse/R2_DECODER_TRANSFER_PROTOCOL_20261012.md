# Small decoder transfer: frozen before candidate completion

The2,048-step run remains active and unchanged. Prepare only CPU-generated
inputs now; inference waits for completePASS and exact selected-weight reload.
No checkpoint, sampling, budget, architecture or temperature selected from this
diagnostic. No repeated candidate evaluations or favorable-arm selection.

Seed179,128previously unused states,8questions per state: four extrema,
two scalar-time order, count and conditional selection. Six paired arms:
base minimum/maximum wording; options reversed; state items reversed;
expanded values7*x+19; domain terms cheapest/priciest,safest/riskiest,
earliest/latest; expanded values plus domain terms. Keep all questions/options,
threshold semantics and target identities. Rendering reuses the prior vocabulary
probe templates; these templates/weaknesses are observed, not novel or blind.
Only state records are new. Verify zero latent-state/normalized-content overlap
with all previous generated train/validation/transfer corpora, oracle labels,
row count6,144,base question count1,024 and128independent state groups.
Repeated count/conditional content across word arms must be counted explicitly.

After selection, compare pinned base SmolLM2, the new validation-selected delta,
and the previous comparison-vocabulary BERT checkpoint step1664
SHA868cc435f3b70ce84d0c49781d80680ba28a1078cbcf6de25f53a58756661b13.
The models differ in pretraining, architecture, readout and inherited training:
no equal-compute or capacity-only causal claim. Use their already-fixed prompt,
processor and scoring pipelines; fail on truncation. Log source/input/model hashes.

Report all per-arm/task Accuracy/NLL/Brier/ECE, condition-pair both-correct,
option/state-order prediction consistency mapped to target identities, and raw
probabilities. For each adapted model versus base, paired128-state bootstrap,
10,000draws,seed17,percentile95% Accuracy intervals; no simultaneous correction.
The direct adapted-decoder versus BERT comparison is descriptive. No aggregate
shortcut to95%, no scalar-time-to-video claim, no sealed audit or product promotion.

Prepared inputs:6,144condition rows,1,024base question IDs,128state groups and
5,632distinct normalized contents. SHA
dc5871692646473f8eac216aa7e0e9ee065040a4c5c9f6e3e22c0a16099ef51d.
No inference has been performed at preparation. Evaluator fails closed unless
all nine scheduled validation records,2,048updates, best selection, reload flag,
model/delta/input hashes and experiment identity pass. It records selection
before loading models. Output directory must be new.

# Fixed semantic adaptation: completed, retention gates failed

Execution **PASS**; research advancement **FAIL**; no product adoption.
One seed17 DeBERTa xsmall fit completed all1024 optimizer calls. The selected
checkpoint is the unchanged pretrained step0 because every trained checkpoint
violates the frozen31-slice contract. Preserve the1024 endpoint for the next
matched native-image diagnosis; do not silently substitute it as the selected model.

## Fixed experiment and actual execution

- Executed source `225a22c134ea42f979443d9cbe6e98a40acb2c8d`.
- [Predeclared protocol](R2_DECISION_SEMANTIC_FIT_PROTOCOL_20261015.md),
  [unchanged config](../../configs/pretrained_reuse/decision_semantic_fit_v1.json).
  Config SHA256 `d9b42743ba5bc4868c5cef2e675da343b86fd78ca69b6f3744609a8e262755af`.
- `MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33`, revision
  `262ae02f29173eec1c250f90804dc7edc677dcff`;70830722 total parameters.
  Freeze49191168 embedding parameters; adapt21639554 parameters in199 encoder,
  pooler and classifier tensors. FP32/eager, dropout off, no added parameters.
- One question's full supplied candidate set per backward,18 questions/update:
  8 binding +8 numeric +2 legacy, each question once. Candidate CE plus coefficient1
  legacy KL to this run's own step0 probabilities; AdamW peak2e-5,31-update warmup,
  cosine to zero. No performance early stopping, seed search or ratio search.
- Main process:2026-10-15 10:39:12–11:49:11 UTC,4199.184s wall time.
  Training2631.623s, evaluation1473.685s; remaining time includes setup/checks.
  Mean training2.570s/update. Local RTX3080Laptop16GB only; cloud cost0.
- Peak CUDA allocation3408770048 bytes (3.175GiB), peak host working set2407784448
  bytes (2.242GiB). Python3.11.9, Torch2.6.0+cu124, Transformers5.6.2,
  NumPy2.4.6, CUDA12.4. No environment upgrade.
- Fresh selected inference took255.722s in a distinct process; no optimizer step.
  These are process wall times, not measured GPU-kernel duty hours.

Commands, from the source checkout with the existing Python3.11 environment:

```powershell
python -u -m scripts.train_decision_semantic --output C:/CodexArtifacts/pretrained-reuse-r2/decision-semantic-fit-01
python -u -m scripts.train_decision_semantic --output C:/CodexArtifacts/pretrained-reuse-r2/decision-semantic-selected-replay-01 --replay-reference C:/CodexArtifacts/pretrained-reuse-r2/decision-semantic-fit-01 --replay-sha256 6994298f92aea987dda2c4c11922dcda6447432ef7c5c661de2749b205bc28df
```

## Learning curves: observed development data

Selection uses8684 questions:3072 new and5612 legacy,31 exact slices. Macro is an
equal-slice mean. Binding/numeric columns are their8-slice means, not native
Image/Video quality. See the [155-row per-slice CSV](R2_DECISION_SEMANTIC_FIT_CURVES_20261015.csv)
for every count, Accuracy/NLL/Brier/ECE/confidence and failed metric.

| Step | New binding Accuracy | New numeric Accuracy | Legacy Accuracy | Macro Accuracy | Macro NLL | Macro Brier | Macro ECE | Best |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|0|24.22%|24.46%|33.39%|28.72%|1.92218|.80725|.14442|0|
|256|100%|47.95%|75.90%|74.91%|.73386|.36696|.16698|0|
|512|100%|68.12%|80.09%|82.14%|.64507|.32155|.19427|0|
|768|100%|72.51%|81.83%|84.12%|.60382|.29210|.19281|0|
|1024|100%|75.59%|82.55%|85.26%|.62058|.29770|.21773|0|

Step1024's minimum across all31 slices is37.8906%, so the aggregate conceals a
weak slice. Its minimum across the16 new slices is53.5156%, versus0 at step0.
The predeclared10000-replicate seed17 whole-state bootstrap gives endpoint gain
95% interval[47.2656,58.5938] percentage points. States are stratified binding128 /
numeric256; all questions are retained. This supports improvement on this
development contract, not a blind transfer/general-quality result.

The selected step0 has zero improvement and interval[0,0]. Research advancement
therefore remains false despite the endpoint's large improvement. Selection is
not changed after seeing the results.

## Retention, calibration and overfitting signals

The frozen rule requires each slice to stay within1pp Accuracy and+.02/.01/.02
NLL/Brier/ECE of step0, then minimizes macro NLL among eligible checkpoints,
including step0. All trained snapshots are ineligible:

| Step | Failed slice/metric comparisons |
|---:|---|
|256|18 ECE;1 Brier|
|512|18 ECE|
|768|18 ECE;1 Brier|
|1024|18 ECE;1 NLL;1 Brier|

The1024 NLL/Brier failures are `legacy:binding:count`: NLL2.63356 versus2.58430,
Brier.97794 versus.88890. There are no Accuracy retention-bound failures, but
that does not satisfy the probability-quality requirement.

At512, numeric ECE.24867 consists mainly of underconfidence (.23127), whereas
legacy binding-count has overconfidence contribution.31730 of ECE.32377.
Calling every ECE failure catastrophic forgetting would misdescribe the evidence.
Binding's1024 new validation answers are all correct from256 onward; step0 had704
true-but-unrequested answers. The improvement covers this limited wording contract.

From768 to1024, macro Accuracy rises while validation NLL/Brier/ECE worsen.
Record this calibration/overfitting warning; it is not evidence that more training
is unconditionally beneficial. Fixed TRAIN has280 questions across35 slices,
with different wording/count distributions from selection. Its descriptive gap
and rolling training loss must not be treated as a clean controlled generalization
gap. Full fixed-TRAIN metrics and every validation prediction are saved locally.

## Consumption and integrity

| Quantity | Entire run | Selected checkpoint0 |
|---|---:|---:|
|Optimizer calls|1024|0|
|Positive-LR calls|1023|0|
|Questions consumed / unique|18432 /18432|0 /0|
|Repeated questions|0|0|
|Unique normalized states|2626|0|
|Native media assets|0|0|

The last18 questions were consumed with zero LR; do not count that last call as
a positive-LR update. Sources:16384 programmatic questions/1536 states,
1024 curriculum questions/128 states,1024 original questions/962 states.
New training contains8192 binding and8192 numeric questions.

The tokenizer preflight and truncation records match byte-for-byte in fresh replay.
No new programmatic row is truncated. The fixed legacy interface truncates only
the state side for48 replay TRAIN and64 legacy validation questions at512 tokens;
no question/candidate is shortened and no row is excluded. This is not a claim
about full-context legacy performance.

All937 pinned source/config files and model/data sources remained unchanged.
Same-process and fresh-process selected validation/fixed-TRAIN scores, metrics and
predictions match exactly (six payload files). The independent CPU verifier also
reconstructs sampler order, all18432 IDs,35 fixed-TRAIN/31 validation slices, every
selector decision, bootstrap intervals, all1024 LR/counters and checkpoint tensor
shapes/dtypes/finiteness. Initial weights match the pinned pretrained tensors;
final weights match step1024. No optimizer/RNG state is saved: no exact-resume claim.

Pre-run Ruff and CPU suite passed1684 tests/1 skip;25 manifest validation commands
passed, source CI#1771 succeeded. Result-publication CI is tracked separately on
its exact commit; these counts are integrity evidence, not quality evidence.

## Artifacts and hashes

Root: `C:/CodexArtifacts/pretrained-reuse-r2/`.

| Artifact | SHA256 |
|---|---|
|`decision-semantic-fit-01/result.json`|`6994298f92aea987dda2c4c11922dcda6447432ef7c5c661de2749b205bc28df`|
|`decision-semantic-selected-replay-01/result.json`|`2e8a8e5cc82e92849484c91064dc716bffe083078a461d0446927adf0d5541f6`|
|`decision-semantic-independent-verification-01.json`|`bf5601f7fa6874bd82ca871e3c153e6e0a0987e0d928266802dfa38597a8bf50`|
|`verify-decision-semantic-v1.py` (executed independent verifier)|`4aaa9c8b75d8cc50a1d3237a6f5ece13cb848d99c2651fced0905cf66682c553`|
|`decision-semantic-calibration-all-01.json`|`6142821e033228814989fa091643d37b85bd360e79c3a56ecd45e727e20bc074`|
|Per-slice CSV|`74c97baa247045e117a69b383e09167f91b4a2918c07d4018a07507c2b4c74d8`|

Corpus manifest `decision-supervision-corpus-01/result.json`:
`f2fac1bb77e762748e9ce49077009bb69a50d7c2869b93c27de6ffbeb7c55d17`.
The fit directory preserves initial,0/256/512/768/1024,best and final deltas,
history, every prediction, environment, effective config, all input/code/artifact
hashes and token records. Each scheduled delta is86625298 bytes. No weight/media
is added to Git. Transfer/audit payloads remain unopened; old artifacts are unchanged.

## Next action

Do not adopt this endpoint, add more synthetic-only training, change the selector
or fit a temperature to these observed validation results. Run the predeclared
[matched native-image diagnosis in the current strategy](NEXT_STRATEGY_20261015.md)
with step0 and the fixed1024 endpoint. Correct annotation facts versus frozen
image-predicted facts identify which remaining errors arise in the decision
interface and which depend on perception. The diagnostic is on previously
observed TRAIN images; it cannot qualify95–98% native quality.

Complete <=200M resident parameters, calibrated native four-modality quality,
temporal/joint benefit, OOD and package/runtime gates remain open.

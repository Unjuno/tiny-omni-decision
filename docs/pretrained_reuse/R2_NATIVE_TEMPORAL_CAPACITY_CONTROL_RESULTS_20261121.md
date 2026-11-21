# R2 native temporal capacity-matched control results

## Outcome

The single seed-17 capacity control completed all 2,048 updates. Its selected
checkpoint is step 1,792; a fresh-process reload reproduced saved predictions
exactly. The LSTM and capacity-matched tokenwise MLP have statistically
indistinguishable Video results on the exposed development validation. This
does not show that recurrence caused the small Video NLL/Brier change seen
against the baseline, and does not justify adopting an LSTM for the product.

The selected capacity control reached 30.08% macro Accuracy and 1.7011 macro
NLL. At step 2,048 these were 29.81% and 1.7073, respectively. The full four
modality target of 95–98% remains unmet by a wide margin. No model was promoted.

## Experimental controls

- Source commit: `9e36b89d84b6a7ff09b7d00061dea2fedc0ea899`.
- Config: [`native_common_capacity_control_v1.json`](../../configs/pretrained_reuse/native_common_capacity_control_v1.json), SHA-256 `d8eb90a067a8c8954cc5e4d53949c4d5f98bb38b955369c6b5581f066ace2739`.
- Same frozen Path A corpus, split, row order, sampler, option shuffling, seed, optimizer, schedule, selector and 2,048-update budget as the baseline/LSTM arms.
- Only architecture change: a tokenwise `192 → 768 → 192` MLP on speech/video tokens in place of recurrent cells. Tokens cannot exchange state. A zero-initialized residual preserves the original baseline function at initialization.
- Added trainable parameters: 1,035,264 for the MLP versus 1,036,416 for the LSTM, a 1,152-parameter difference (0.111%). Total model size is 197,290,138 parameters; it remains below the provisional 200M parameter ceiling. Package size and product deployment quality were not measured.
- Checkpoint selection: lowest validation macro NLL, `min_delta=1e-5`, earliest checkpoint on ties. All arms used their existing selected checkpoint for the selected table below; final means update 2,048.
- The 24,346-question development validation was already exposed in the earlier comparison. It is exploratory development evidence, not a blind test. `sealed_audit_loaded=false`; no sealed audit was opened.

The sampler consumed 8,192 training examples: 2,048 per modality. This included
6,656 unique questions, 1,536 repeated question draws, 5,107 unique observations,
3,114 unique media assets, and 3,382 groups. Source counts were Clevr-4 2,048,
CLEVRER 2,048, Lombard 1,024, STARSS 1,024, condition-relevance text 683,
programmatic text 682, curriculum text 497, and original text 186. The fixed
TRAIN probe contained 1,456 examples; validation contained 24,346.

## Run and artifact integrity

The first attempt, `fit-01`, terminated before any optimizer update when Windows
reported `STATUS_IN_PAGE_ERROR` (`0xC0000006`, Application Error 1005) reading
the pinned `D:\CodexArchiveRelocated\CodexArtifacts\pretrained-reuse-r2\native-language-cache-02\train-states.npy`.
Its receipt remained stale `RUNNING`; it is not counted as a result. Inputs
were not changed. The identical frozen run was started from a new output path
after the drive became available.

| Artifact | Status | SHA-256 |
|---|---|---|
| Smoke receipt (`smoke-01`) | `PASS_NATIVE_FIT_SMOKE` | `92d452fafe9afe599d0106a832fccd8d0fd2b3d2e0578ccc658b5cdfd61c8573` |
| Fit receipt (`fit-02`) | `PASS_NATIVE_FIT_TRAIN`, 2,048 updates, best step 1,792 | `348965b1c0dbaa0e8535eca07cc5d58644c06c6a48113e61834510702c336db3` |
| Replay receipt (`replay-01`) | `PASS_NATIVE_FIT_REPLAY`, maximum prediction delta 0 | `ae663838b88e53a2aaf862a7299a5289d5b495cc8d72bffe40a2c29249d867dd` |
| Comparison JSON (`comparison-02`) | `PASS_NATIVE_TEMPORAL_CAPACITY_COMPARISON` | `21a4a4b2a1e071d4ebd17c805a65f9170eed7b6aa74b6a78dd2fce8b6fba8fe5` |

Frozen data pins include corpus receipt `47a6d2f3495810bac58f57dec1a962f8b490fa409c75671b46bb2999fde87d49`, train JSONL `d279ca44c6ff8e619529805be21675a8b4088fdcc476c7bb04649abdb5e60ebb`, validation JSONL `1f545a6e8d10b2712980f9ada58a0e5eef5b21021e4e01d6483965889a970326`, query-cache receipt `7f5ffd7362e84770b864a189d50bc29f43be3738f54a07f6dbdc21f1ef7d909e`, and query array `a453b982e5d739303779b6dfccc91537d3521e8ad6be490263b44623c7f1f7e7`. Validation-ID and sampler hashes are recorded in the fit receipt as `db7d0f1d09319a48a7725ed36ccb515cdcc85a409f019400cf0062be38768f30` and `b2647f3a1783fe5909ab4864281522314b694ef9a33851a35cc11b6beabe41fb`.

Run outputs, checkpoints, prediction arrays, and the comparison JSON remain in
`C:\CodexArtifacts\pretrained-reuse-r2`; no weights, media, or large artifacts
were added to Git. The fit ran locally on the RTX 3080 Laptop GPU with zero
cloud cost. Fit wall time was 6,219.8 s (103.7 min), of which 1,826.8 s was
training and 3,463.6 s validation evaluation. Peak CUDA allocation was
256,556,544 bytes (reserved 358,612,992 bytes); peak process working set was
3,479,171,072 bytes. Replay wall time was 1,645.2 s and also used no cloud
compute.

The run evaluated every 256 updates. Step 1,920 was not scheduled, despite
being listed in the original comparison request. The corrected comparison
reports the common saved points 512, 768, 1,024, 1,280, 1,536, 1,792 and 2,048;
it does not interpolate or synthesize step 1,920. The analysis-only fix was
made after fit/replay: current comparison-script SHA-256 is
`a2a5782a6ccce87c628e91681aea10134bd741da8c9f8af229aa0d8136a2b93f`, whereas
the fit receipt pins the earlier script at
`44dbc70a054bb6a1ddb544194335c5b7921413fafb355c12d4df4b5234a39738`. Neither
fit nor replay receipt/model artifacts was edited. The schedule helper has two
focused passing tests, and Ruff passes on the changed analysis code/tests.

## Learning curve

Selected checkpoint is chosen by validation macro NLL; final is always step
2,048. Values below are the capacity MLP arm.

| Step | Macro Accuracy | Macro NLL | Macro Brier | Macro ECE | Video Accuracy | Video NLL |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 0.2774 | 1.7847 | 0.7609 | 0.0915 | 0.3150 | 1.3572 |
| 768 | 0.2824 | 1.7785 | 0.7648 | 0.1005 | 0.3300 | 1.4455 |
| 1,024 | 0.2898 | 1.7732 | 0.7546 | 0.0866 | 0.3225 | 1.3451 |
| 1,280 | 0.2808 | 1.7214 | 0.7470 | 0.0696 | 0.3200 | 1.3427 |
| 1,536 | 0.2878 | 1.7028 | 0.7447 | 0.0744 | 0.3200 | 1.3309 |
| 1,792 (selected) | 0.3008 | 1.7011 | 0.7434 | 0.0777 | 0.3300 | 1.3311 |
| 2,048 (final) | 0.2981 | 1.7073 | 0.7450 | 0.0805 | 0.3250 | 1.3359 |

The selected-to-final macro NLL worsened by 0.0061 and macro Accuracy fell by
0.27 percentage points. The final scheduled validation also emitted the
existing warning that train-probe NLL continued falling while validation macro
NLL rose. This is a modest late-overfit signal, not a capacity-limit result.
The external comparison JSON contains the baseline and LSTM curves as well.

## Validation metrics

The macro table reports the hierarchical macro metric used by the existing
selector. Modality rows report task-macro Accuracy/NLL/Brier/ECE.

| Arm | Checkpoint | Macro Accuracy | Macro NLL | Macro Brier | Macro ECE | Minimum modality Accuracy |
|---|---|---:|---:|---:|---:|---:|
| Baseline | selected 1,792 | 0.3071 | 1.6865 | 0.7402 | 0.0842 | 0.1831 |
| LSTM | selected 1,536 | 0.3019 | 1.6967 | 0.7423 | 0.0750 | 0.1470 |
| Capacity MLP | selected 1,792 | 0.3008 | 1.7011 | 0.7434 | 0.0777 | 0.1724 |
| Baseline | final 2,048 | 0.3058 | 1.6910 | 0.7415 | 0.0847 | 0.1807 |
| LSTM | final 2,048 | 0.3071 | 1.7071 | 0.7444 | 0.0779 | 0.1782 |
| Capacity MLP | final 2,048 | 0.2981 | 1.7073 | 0.7450 | 0.0805 | 0.1685 |

Selected-checkpoint modality metrics:

| Arm | Modality | Accuracy | NLL | Brier | ECE |
|---|---|---:|---:|---:|---:|
| Baseline | Audio | 0.5292 | 1.1761 | 0.5478 | 0.0879 |
| Baseline | Image | 0.2012 | 2.1651 | 0.8652 | 0.0722 |
| Baseline | Text | 0.1831 | 2.0656 | 0.8471 | 0.0889 |
| Baseline | Video | 0.3150 | 1.3393 | 0.7006 | 0.0880 |
| LSTM | Audio | 0.5184 | 1.1497 | 0.5421 | 0.1075 |
| LSTM | Image | 0.1470 | 2.2213 | 0.8794 | 0.0469 |
| LSTM | Text | 0.1845 | 2.0818 | 0.8507 | 0.0893 |
| LSTM | Video | 0.3575 | 1.3339 | 0.6970 | 0.0562 |
| Capacity MLP | Audio | 0.5153 | 1.1817 | 0.5542 | 0.0871 |
| Capacity MLP | Image | 0.1724 | 2.2213 | 0.8787 | 0.0577 |
| Capacity MLP | Text | 0.1857 | 2.0706 | 0.8455 | 0.0867 |
| Capacity MLP | Video | 0.3300 | 1.3311 | 0.6953 | 0.0793 |

Final-checkpoint modality metrics:

| Arm | Modality | Accuracy | NLL | Brier | ECE |
|---|---|---:|---:|---:|---:|
| Baseline | Audio | 0.5303 | 1.1854 | 0.5494 | 0.0909 |
| Baseline | Image | 0.1973 | 2.1623 | 0.8646 | 0.0608 |
| Baseline | Text | 0.1807 | 2.0720 | 0.8486 | 0.0952 |
| Baseline | Video | 0.3150 | 1.3444 | 0.7035 | 0.0920 |
| LSTM | Audio | 0.5317 | 1.1900 | 0.5537 | 0.0970 |
| LSTM | Image | 0.1782 | 2.2004 | 0.8744 | 0.0598 |
| LSTM | Text | 0.1859 | 2.1004 | 0.8498 | 0.0938 |
| LSTM | Video | 0.3325 | 1.3375 | 0.6995 | 0.0609 |
| Capacity MLP | Audio | 0.5153 | 1.1919 | 0.5564 | 0.0871 |
| Capacity MLP | Image | 0.1685 | 2.2175 | 0.8776 | 0.0523 |
| Capacity MLP | Text | 0.1835 | 2.0839 | 0.8474 | 0.0920 |
| Capacity MLP | Video | 0.3250 | 1.3359 | 0.6987 | 0.0905 |

## Paired Video uncertainty

Effects are capacity MLP minus the comparator on the fixed final (step 2,048)
predictions. Intervals use 5,000 paired scene-cluster bootstrap resamples, seed
17, and percentile 95% intervals with NumPy linear quantiles. These intervals
do not include training-seed or checkpoint-selection uncertainty.

| Contrast / subset | Questions / scenes | Accuracy delta (95% CI) | NLL delta (95% CI) | Brier delta (95% CI) |
|---|---:|---:|---:|---:|
| MLP − baseline, all Video | 400 / 200 | +0.0100 (−0.0200, +0.0401) | −0.00857 (−0.01502, −0.00225) | −0.00486 (−0.00859, −0.00125) |
| MLP − baseline, temporal descriptive | 200 / 200 | +0.0100 (−0.0300, +0.0500) | −0.01017 (−0.02172, +0.00085) | −0.00503 (−0.01110, +0.00105) |
| MLP − LSTM, all Video | 400 / 200 | −0.0075 (−0.0325, +0.0175) | −0.00160 (−0.00686, +0.00364) | −0.00079 (−0.00431, +0.00275) |
| MLP − LSTM, temporal descriptive | 200 / 200 | +0.0100 (−0.0250, +0.0450) | −0.00139 (−0.00953, +0.00657) | −0.00104 (−0.00676, +0.00446) |

Against baseline, the MLP point estimate improves Video NLL/Brier over all
Video questions, with intervals excluding zero, but its Accuracy interval
includes zero. In the temporal-descriptive subset all three intervals include
zero. Against the LSTM, all intervals include zero in both subsets. The MLP
therefore reproduces similar Video NLL/Brier without recurrence, but these
single-seed, validation-selected comparisons cannot establish that capacity
alone caused the result or that recurrence has no value on other tasks.

## Decision and next work

- Do not expand to xLSTM or spend more training budget on LSTM rank/capacity sweeps based on this result.
- Keep the original baseline as the macro-validation reference; retain LSTM and MLP receipts as diagnostic arms.
- Continue with task-specific temporal/input-use diagnostics and improve the weak Image/Text paths before another general architecture change. Any new fit needs a fresh, preregistered evaluation source or train-only grouped folds; this exposed validation must not be treated as blind.
- All four modality quality targets, real paired/joint-input benefit, calibration on untouched evaluation data, and an end-to-end product package/runtime gate remain open.

The full receipt-rich comparison is stored at
`C:\CodexArtifacts\pretrained-reuse-r2\native-common-capacity-control-comparison-02\comparison.json`
(SHA-256 `21a4a4b2a1e071d4ebd17c805a65f9170eed7b6aa74b6a78dd2fce8b6fba8fe5`).

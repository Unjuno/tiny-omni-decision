# QASC same-state / new-criterion diagnostic results (2026-11-05)

## Outcome

**The preregistered, narrow state-dependence rule is supported on this internal QASC TRAIN diagnostic.** A fixed seed-17 `FusionDecisionScorer` was fit once on 7,451 rows and evaluated once after epoch 8 on 683 held rows. The held questions were globally new to this fit, while each of the 666 fact-pair states remained represented in training through other questions.

| Held condition (same questions/options/targets) | n | Accuracy | NLL | Brier | ECE-15 | Mean confidence |
|---|---:|---:|---:|---:|---:|---:|
| Correct state + held criterion | 683 | 0.7862 | 0.7065 | 0.3205 | 0.0653 | 0.8406 |
| Correct state + criterion absent | 683 | 0.7013 | 1.1174 | 0.4605 | 0.1378 | 0.8289 |
| Zero state-content + criterion | 683 | 0.1537 | 4.7872 | 1.2745 | 0.5437 | 0.6975 |
| Different deranged state + criterion | 683 | 0.1596 | 5.7429 | 1.3608 | 0.5986 | 0.7582 |
| Uniform 8-choice control | 683 | 0.1274 | 2.0794 | 0.8750 | 0.0024 | 0.1250 |
| Training answer-position prior | 683 | 0.1069 | 2.0800 | 0.8751 | 0.0233 | 0.1302 |

The predeclared paired state-cluster bootstrap used 5,000 resamples (seed 1718, percentile 95% intervals). Differences below are `correct state + criterion` minus the control; favorable Accuracy intervals are positive and favorable NLL intervals are negative.

| Control | Accuracy difference (95% CI) | NLL difference (95% CI) | Brier difference (95% CI) |
|---|---:|---:|---:|
| Zero state-content | +0.6327 (+0.5912, +0.6720) | −4.0821 (−4.3833, −3.7928) | −0.9542 (−1.0145, −0.8937) |
| Deranged different state | +0.6269 (+0.5873, +0.6672) | −5.0368 (−5.3731, −4.6988) | −1.0405 (−1.1017, −0.9786) |
| Criterion absent | +0.0849 (+0.0534, +0.1162) | −0.4096 (−0.5181, −0.3074) | −0.1397 (−0.1771, −0.1041) |

This meets the fixed rule for state-content contribution: correct-state Accuracy is higher and NLL lower than both zero-state and shuffled-state controls, with paired group intervals excluding zero. The held criterion also contributes relative to the same checkpoint with criterion absent. This supports a narrow within-state transfer result for this QASC setup; it does not establish general reasoning or multimodal quality.

## Shortcut, calibration, and limits

The normalized target option text occurs in the supplied evidence in 542/683 held rows (79.36%). On those rows, correct-state Accuracy/NLL/Brier/ECE-15 were 0.8026/0.6622/0.3013/0.0660. On the 141 rows where that target string does not occur in either fact, the corresponding values were 0.7234/0.8770/0.3943/0.1194. The latter subset is small, and no separate interval was preregistered for that stratum. Exact target-string leakage is therefore not the only plausible signal, but lexical and dataset-specific shortcuts remain serious alternatives.

The correct-state model is still overconfident: mean confidence 0.8406 versus 0.7862 Accuracy, ECE-15 0.0653. Zero-state and shuffled-state controls are also severely overconfident despite near-chance Accuracy. A deployed system must not interpret a score from absent or mismatched evidence as a reliable decision; this experiment did not implement an abstention policy.

### Post-hoc target-not-in-evidence sensitivity (2026-11-05)

After the fixed aggregate result, the already saved predictions were stratified on the protocol's `target_in_evidence` field. No model, data, prediction, or checkpoint was changed. On the 141 held rows (141 distinct state groups) where the exact normalized target string was absent from both supplied facts, the correct-state + criterion condition scored 102/141 (72.34%) Accuracy, 0.8770 NLL, 0.3943 Brier, and 0.1194 ECE-15. Same-checkpoint controls scored:

| Condition | Accuracy | NLL | Brier | ECE-15 |
|---|---:|---:|---:|---:|
| Correct state + criterion | 0.7234 | 0.8770 | 0.3943 | 0.1194 |
| Correct state, criterion absent | 0.6312 | 1.3518 | 0.5337 | 0.1656 |
| Zero state content + criterion | 0.2128 | 4.2001 | 1.1673 | 0.4864 |
| Deranged different state + criterion | 0.1560 | 5.4280 | 1.3508 | 0.6005 |

Exploratory paired state-cluster bootstrap (5,000 percentile resamples, seed 1718) intervals for correct-state minus control were +9.12 pp Accuracy (+2.13, +16.31), -0.4762 NLL (-0.7032, -0.2665), and -0.1387 Brier (-0.2176, -0.0648) against criterion-absent; +50.97 pp Accuracy (+41.84, +59.57), -3.3183 NLL (-3.9457, -2.7020), and -0.7716 Brier (-0.9005, -0.6467) against zero-state; and +56.67 pp Accuracy (+47.52, +65.96), -4.5467 NLL (-5.2437, -3.8668), and -0.9550 Brier (-1.0855, -0.8225) against the deranged state. This weakens the specific explanation that the observed signal comes solely from copying the exact target string, but does not rule out semantic/lexical shortcuts, dataset artifacts, or exposure to all QASC TRAIN rows in earlier project experiments. The stratum and this follow-up were post-hoc, remain development-only, and were not used for model selection or promotion. Reproduction command: `$env:PYTHONPATH='src'; py -3.11 scripts/analyze_qasc_same_state_nonliteral.py --run-dir 'C:\CodexArtifacts\pretrained-reuse-r2\qasc-same-state-new-criterion-20261105-attempt-01'`. The analysis script SHA-256 is `2b861d539d7f31e292060fb0ec921752f978a0c5fcf164500001077f161323b6`; machine-readable output is `C:\CodexArtifacts\pretrained-reuse-r2\qasc-same-state-new-criterion-20261105-attempt-01\target-not-in-evidence-analysis.json` (SHA-256 `8fe32f0e07d8633446b32dfb405082870d3b4191e447f475a93c5a9d5087092b`). The input held-prediction file remains SHA-256 `7138c49afb424e615aa1c95b08cc0e733368010f4ed49ab18ffaad49a6de9650`.

Every QASC TRAIN row was already in the universe of the preceding five-fold QASC experiment. These held questions are new only relative to this one fixed fit; they are **not project-level unseen data or independent quality evidence**. QASC official validation, test, and sealed audit were not read. This is text only and does not demonstrate persistent cache/runtime savings, video/audio/image/sensor competence, or M2 cross-modal fusion. Keep R2-M and product promotion gates open.

## Frozen split, model, and training

- Source: `allenai/qasc@a34ba204eb9a33b919c10cc08f4f1c8dae5ec070`, CC BY 4.0; frozen prepared TRAIN SHA-256 `6ad3536ebba70a35f87bf078d0d2837df09fda652201a2b4a6fd7d7b01ed7f0b`.
- 8,134 source TRAIN rows; 7,139 normalized fact-pair states; 673 states have at least two distinct questions; 666 satisfy the globally unique held-question rule. Resulting split: 7,451 train / 683 held / 666 states.
- Intentional state overlap; every held state appears in training. Normalized held-question overlap is 0; normalized question plus option-set overlap is 0.
- Ordered train-ID SHA-256: `6d7408b3cf5bb58a6dabe753ed88d90a744645564af60b87df1d144c4bb460cf`. Ordered held-ID SHA-256: `d44f521df10ddee02e026613438c30453eb4c89ccca5683a9e687062ab2f1918`.
- Encoder: frozen MiniLM-L3 revision `4ca70771034acceecb2e72475f72050fcdde4ddc`; weight SHA-256 `cf1e4e2d420c664973037c3c73125d7a8fc69952495093ef8f50596f8943a433`. The runtime also checks the existing text readout SHA-256 `12d8278cb0c13c384216ba706c4edd39a30ce6888f6a1900ea7ce26304bf02d0`.
- Readout: `FusionDecisionScorer`, width 128, 164,610 trainable parameters; fresh initialization, seed 17; AdamW `lr=0.001`, weight decay `0.0001`, batch 64, 8 epochs. The final epoch-8 checkpoint was fixed in advance; the held set was not scored during fitting or used for checkpoint selection.
- Training presentations: 59,608 (7,451 unique training rows repeated across the fixed eight epochs), 936 optimizer updates. Rolling train CE by epoch: 1.900, 1.334, 0.926, 0.690, 0.521, 0.397, 0.297, 0.219.
- `final-epoch-8.pt` is 661,640 bytes. Encoder plus scorer plus the existing Path A text readout totals 17,586,689 resident parameters; only the 164,610 scorer parameters were trained. This is an artifact measurement, not a product package or size qualification.

## Runtime and integrity

- Run directory (outside Git): `C:\CodexArtifacts\pretrained-reuse-r2\qasc-same-state-new-criterion-20261105-attempt-01`.
- Source commit `0858f851ea4e7781880c3f46c5634e4e1a23d251`; experiment config SHA-256 `c9310fd44b88cefbe843845f10fdbdd4ee31dcffe8070231143785ee8549b88c`; effective config SHA-256 `a6723b896101529122dec440a8490b9da45cad12ac68ec3dd03e25133104552b`.
- Role-feature matrix SHA-256 `f9b8e50929a91352cde0ec32d3c9f6982ca42a33e28b071d1f2f3ab36e44fb7e`; checkpoint SHA-256 `6407ad1c2e24ff099332f7c9c347baec7f4e6daee721b82dcf9ca56f18db0e20`; held predictions SHA-256 `7138c49afb424e615aa1c95b08cc0e733368010f4ed49ab18ffaad49a6de9650`; full `report.json` SHA-256 `a0ee840d284fdcbfa09877b4e32bcb0f3ce2636a9abbd873fc04f1283a271521`.
- Run: 2026-11-05 01:04:13–01:06:40 JST; 8.05 seconds fitting, 0.48 seconds final metrics/reload checks, 147.75 seconds from feature encoding through final report. The role features were computed on CPU. A sampled process working set during encoding was about 1.01 GB; it was not a peak-memory measurement.
- Hardware/software: Intel Core i7-12700H; Python 3.11.9; PyTorch 2.6.0+cu124; Transformers 5.6.2. `CUDA_VISIBLE_DEVICES=-1`; explicit CPU runtime. No GPU memory was used. At preflight the separate Orbitune PID 17132 held 15,832 / 16,384 MiB on the local RTX 3080 Laptop. No process was stopped or changed. No cloud compute; cost $0.
- Save/reload probabilities matched exactly; reversed option-order probabilities realigned within `1.1921e-7` maximum absolute difference (tolerance `1e-6`). The state derangement had 666 groups and 0 fixed points.
- Output directory totals 2,308,161 bytes; no model, media, feature matrix, or prediction artifact was added to Git.

Exact invocation (PowerShell):

```powershell
$env:PYTHONPATH = 'src'
$env:CUDA_VISIBLE_DEVICES = '-1'
py -3.11 scripts/run_qasc_same_state_criterion.py `
  --train-jsonl 'C:\CodexArtifacts\pretrained-reuse-r2\qasc-evidence-context-20261104-attempt-01\prepared\evidence-train.jsonl' `
  --model-dir 'C:\CodexArtifacts\pretrained-reuse-r2\path-a-modular-runtime-assembly-v4\components\audio\models\text-minilm-en' `
  --text-readout 'C:\CodexArtifacts\pretrained-reuse-r2\path-a-modular-runtime-assembly-v4\components\text\readout\best-readout.pt' `
  --output-dir 'C:\CodexArtifacts\pretrained-reuse-r2\qasc-same-state-new-criterion-20261105-attempt-01'
```

## Verification and next gate

Focused QASC tests: 20 passed. Full CPU suite: **2,928 passed, 11 skipped** in 43.14 seconds. Ruff and `git diff --check` pass. Commit `0858f851` is pushed to the existing Draft PR #45 branch; hosted CI was still running at the last check. Do not promote this scorer or begin another QASC budget/seed sweep. The next quality gate remains an admitted synchronized multi-input task with rights, independent group splits, and matched unimodal/shuffled/missing controls.

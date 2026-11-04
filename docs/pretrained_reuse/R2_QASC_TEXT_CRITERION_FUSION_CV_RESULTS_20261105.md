# QASC text-criterion FusionDecisionScorer CV results (2026-11-05)

## Outcome

The fixed five-fold TRAIN-only run completed on CPU in a fresh external output
directory. It tested whether a small `FusionDecisionScorer` can use a supplied
evidence state, natural-language question criterion, and separately encoded
options on one narrow eight-choice science-question task. This is not broad
text quality or multimodal evidence.

| OOF condition | n | Accuracy | NLL | Brier | ECE-15 | Mean confidence |
|---|---:|---:|---:|---:|---:|---:|
| State + criterion + options | 8,134 | 0.6901 | 0.9427 | 0.4362 | 0.0967 | 0.7867 |
| Same checkpoint, criterion masked | 8,134 | 0.6172 | 1.3407 | 0.5594 | 0.1802 | 0.7974 |
| Uniform 8-option control | 8,134 | 0.1242 | 2.0794 | 0.8750 | 0.0008 | 0.1250 |
| Fold-training answer-position prior | 8,134 | 0.1290 | 2.0791 | 0.8749 | ~0.0000 | 0.1290 |

The criterion-present readout exceeds the same-checkpoint criterion-masked
control by 7.29 percentage points Accuracy and lowers NLL by 0.3980. The
criterion-masked result remains far above uniform and answer-position controls,
so the state/options path also carries substantial signal. The full readout is
still overconfident (0.7867 mean confidence vs 0.6901 Accuracy; ECE-15 0.0967).

| Held fold | n | Accuracy | NLL | Brier | ECE-15 | Criterion-masked Accuracy |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1,627 | 0.6902 | 0.9412 | 0.4365 | 0.0961 | 0.6048 |
| 1 | 1,627 | 0.6829 | 0.9888 | 0.4504 | 0.1055 | 0.6085 |
| 2 | 1,626 | 0.6925 | 0.9507 | 0.4333 | 0.0949 | 0.6199 |
| 3 | 1,627 | 0.6976 | 0.9052 | 0.4251 | 0.0937 | 0.6183 |
| 4 | 1,627 | 0.6872 | 0.9278 | 0.4356 | 0.0978 | 0.6343 |

The earlier QASC evidence-context readout scored 0.7799 OOF Accuracy and
0.6598 NLL on the same preserved TRAIN folds; the earlier question-only
readout scored 0.3535 and 1.9197. These are useful references, not a clean
one-factor comparison: this run changes the readout and explicitly separates
the state, criterion, and options. It does not show that this decomposition is
better than the earlier evidence-context formulation. Exact target-option text
occurs in the supplied support facts for 6,167/8,134 rows (75.82%), so shortcut
risk remains high.

## Frozen run and integrity

- Experiment: seed 17, five existing grouped folds, eight epochs per fold,
  batch size 64, AdamW at `1e-3` with `1e-4` weight decay, fixed epoch-8
  endpoints. No held-fold checkpoint selection or tuning was performed.
- Inputs: 8,134 official QASC TRAIN rows from the pinned CC-BY-4.0 corpus;
  preserved fold assignment hash matched exactly. Each row was held out once;
  all five folds had zero fact-pair, normalized-question, and
  question/choice-content overlap.
- Data SHA-256:
  `6ad3536ebba70a35f87bf078d0d2837df09fda652201a2b4a6fd7d7b01ed7f0b`.
- Fold assignment SHA-256:
  `72282c4b57fb2e5a56c4d2f15f75d11bf11f79071d8ac8a981b1a7311bcd0b67`.
- Encoder: `sentence-transformers/paraphrase-MiniLM-L3-v2` revision
  `4ca70771034acceecb2e72475f72050fcdde4ddc`, weights SHA-256
  `cf1e4e2d420c664973037c3c73125d7a8fc69952495093ef8f50596f8943a433`.
- Readout: 164,610 trainable parameters. The frozen text encoder has
  17,389,824 parameters; the loaded Path A text runtime contains 17,586,689
  parameters including its existing text readout, which this fit does not use.
- Training consumed 260,288 row presentations over 4,080 optimizer updates.
  The runner reports 36.51 seconds of fitting, 1.22 seconds of fold evaluation
  and save/reload checks, and 181.39 seconds wall time including model load,
  feature encoding, and serialization.
- Environment: Windows, Python 3.11.9, PyTorch 2.6.0+cu124, Intel Core
  i7-12700H (14 cores / 20 logical processors). `CUDA_VISIBLE_DEVICES=-1`
  was set and the runner hard-codes `device=cpu`; the RTX 3080 was occupied by
  an unrelated process and was not used. Cloud/GPU cost: $0.
- Began `2026-11-05 00:18:48 JST`, completed by `00:21:57 JST`.
- Source commit captured by the runner: `ca6a0e78a26f33e017084c5a8c37b644f4a95e5e`.
  Experiment-config SHA-256:
  `80ccc1cfe6f1c3843367e66cc07d61243678ae29e6aa9024d5f31a13b98ac7e9`.
- Effective-config SHA-256:
  `33559b85f0e7127f7ca57169a412eb1c787037819941c329a99f5852022d35e1`.
- Role feature cache SHA-256:
  `f9b8e50929a91352cde0ec32d3c9f6982ca42a33e28b071d1f2f3ab36e44fb7e`.
- Runner SHA-256: `479c26bf84b5bfb13f8193d0a3fc1e3ed541e1794a7155bcb6da1107b62c0d0c`.
- Feature/metric helper SHA-256:
  `5373c4787adad68f52dcd88a21f5a834eec11e8a3211e02c65c80410aee3b155`.
- Save/reload predictions matched exactly for every fold. Option permutation
  passed in all folds; maximum probability difference was `2.3841858e-7`, below
  the preregistered `1e-6` absolute/relative tolerance.
- The official QASC validation shard, test shard, and sealed audit were not
  read. No model selection used them.

Exact invocation (PowerShell):

```powershell
$env:PYTHONPATH='src'
$env:CUDA_VISIBLE_DEVICES='-1'
py -3.11 scripts/run_qasc_text_criterion_fusion_cv.py `
  --train-jsonl 'C:\CodexArtifacts\pretrained-reuse-r2\qasc-evidence-context-20261104-attempt-01\prepared\evidence-train.jsonl' `
  --fold-assignment 'C:\CodexArtifacts\pretrained-reuse-r2\qasc-path-a-train-only-group-cv-20261104-attempt-01\fold-assignment.jsonl' `
  --model-dir 'C:\CodexArtifacts\pretrained-reuse-r2\path-a-modular-runtime-assembly-v4\components\audio\models\text-minilm-en' `
  --text-readout 'C:\CodexArtifacts\pretrained-reuse-r2\path-a-modular-runtime-assembly-v4\components\text\readout\best-readout.pt' `
  --output-dir 'C:\CodexArtifacts\pretrained-reuse-r2\qasc-text-criterion-fusion-cv-20261105-attempt-02'
```

## Preserved artifacts

External attempt directory:
`C:\CodexArtifacts\pretrained-reuse-r2\qasc-text-criterion-fusion-cv-20261105-attempt-02`.
It contains the effective and experiment configs, exact fold plan and mapping,
five epoch-8 checkpoints, all OOF predictions, and the JSON report. No model,
data, or prediction artifact was added to Git.

| Artifact | SHA-256 |
|---|---|
| `effective-config.json` | `33559b85f0e7127f7ca57169a412eb1c787037819941c329a99f5852022d35e1` |
| `experiment-config.json` | `80ccc1cfe6f1c3843367e66cc07d61243678ae29e6aa9024d5f31a13b98ac7e9` |
| `fold-assignment.jsonl` | `120b5119aab3ed662ebcf09bf2cb6585528a29c6c2c1bbfcde6aaacd945784e1` |
| `fold-plan.json` | `f7ada0535ed04cc5170eea2ae971c52864f9516c63997bba0beb12cebbd98921` |
| `report.json` | `6c16b311dab8633d15c69469264895c0d95c7459d4fc763aafe9b502d7d925a5` |
| `out-of-fold-predictions.jsonl` | `966d90d255b187ecf5208720bf46a2a8d1d362387bb612d6744f335a69d5c317` |
| `fold-0-epoch-8.pt` | `57c4229770d55542b5fa56005bdfbc421cd2d2166d059fe4cfd96fe13a3f49db` |
| `fold-1-epoch-8.pt` | `8d6227625006ebd38dfa59b21b051c3b8b9e6591c8cba25a925597d11b757df8` |
| `fold-2-epoch-8.pt` | `a6bfb20f66d910ba27a671dbf8e0d45416e1d1e2f9f1b557dd1cddc510a44f2f` |
| `fold-3-epoch-8.pt` | `74cc404588c7a2eff0287da7765cc7ec49a1671d445e410fcf0a2ad7b3dcc4ce` |
| `fold-4-epoch-8.pt` | `62db832b6bd2237352a2ae953dcef29b75aeca05582af875e3f82c3f9f6f7f6d` |

Attempt 01 remains preserved separately as a stopped partial run. Its only
failure was the `1e-7` float32 option-order assertion; replay showed a maximum
`2.3841858e-7` delta and no top-option changes. The corrected `failure.json`
records the exact original config hash. Attempt 02 repeats the preregistered
condition from fresh per-fold initialization with the corrected numeric
tolerance; no result from attempt 01 selected a checkpoint or changed the
training schedule.

## Interpretation and next gate

This run supports that the task-specific scorer can use an independently
provided question criterion on QASC TRAIN folds. It also shows a material
quality/calibration gap from the prior evidence-context readout and cannot
disentangle the feature-role design from the changed scorer. The 75.82% target
string-in-evidence rate limits the strength of any generalization claim. Do not
repeat seeds, extend training, reopen validation/test/audit, or promote the
scorer based on this result. R2 joint multimodal task evidence and R3 admission
remain open.

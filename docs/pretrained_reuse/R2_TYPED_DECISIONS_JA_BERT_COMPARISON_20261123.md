# R2 Japanese typed-decision BERT comparison — 2026-11-23

## Finding

For a frozen Japanese encoder plus a small shared candidate scorer, the
111M-parameter Japanese BERT-base reference matched this internal typed
decision benchmark better than the 27.6M compact LightSPLADE-derived encoder.
On the fixed validation IDs, BERT improved macro Accuracy by 10.0 percentage
points and also improved soft NLL and Brier. Its ECE difference is uncertain.
This is evidence to keep investigating Japanese BERT-family text encoders,
especially compact distilled variants, but it is **not** a 95–98% product
result or human-ground-truth benchmark. The labels are distributions from an
English-state teacher model, and checkpoint selection used this validation.

The experiment is a text-only frozen-feature Path A probe. It does not show
multimodal fusion quality, does not alter any encoder weights, and does not
replace the existing Ruri path in the product architecture.

## Data and split

Source: [`GeneLab/typed-decisions-ja`](https://huggingface.co/datasets/GeneLab/typed-decisions-ja),
revision `01a4558e83bf5a748094239df843d658b33d2533`, declared Apache-2.0.
Only the publisher's `train-00000-of-00001.parquet` was downloaded and loaded;
the publisher's public `test` partition was not opened. The source train file
has 1,200 state rows across four workflows and five typed questions per row.
The fixed seed-17 state-group split uses 960 rows / 4,800 decisions for
training and 240 rows / 1,200 decisions for validation, with 60 validation
states per workflow. All five questions from a state remain together. The
split was stratified by workflow, has zero state-group overlap, and has zero
NFKC-normalized canonical state-JSON overlap. All workflow schemas appear in
both partitions, so this measures within-workflow state generalization, not
workflow OOD generalization.

The train parquet SHA-256 is
`14589163a2278225926e599a24f5620902bbbdc998946d1f63fc716a4477be1c`; the
frozen assignment JSONL SHA-256 is
`5873bb0b94d77cbad23aed7418a1a5d9cdf31b18a573434ff62d98fc20297c06`; the
source audit manifest SHA-256 is
`3266d1c07c8312ee48181fd53752abe6d80724a5afb77cce6b1c0257485ac9ad`.

**Label limitation:** the Japanese target probabilities are teacher outputs
based on English-state predictions, not independently adjudicated truth. The
dataset card describes a roughly 4B-parameter teacher sampled three times and
reports 65 back-translation mismatches among 1,200 training states. We retained
all train rows. Metrics below mean agreement with the teacher's modal label or
probability distribution; they do not establish real-world decision accuracy.
No published leaderboard or test-set result was used.

## Frozen encoders and fixed scorer

| Candidate | Pinned source | Encoder parameters used | Readout parameters | Total before quantization | Upstream checkpoint bytes |
|---|---|---:|---:|---:|---:|
| Tohoku Japanese BERT-base v3 | `tohoku-nlp/bert-base-japanese-v3`, revision `65243d6e5629b969c77309f217bd7b1a79d43c7e` | 110,616,576 | 98,561 | 110,715,137 | 447,406,217 |
| LightSPLADE Japanese 28M-derived encoder | `bizreach-inc/light-splade-japanese-28M`, revision `64d8f31a2196074dbd5de7707bbe6f14be3ab985` | 27,566,592 | 49,409 | 27,616,001 | 111,007,808 |

Both runs used the same Japanese BERT tokenizer, pair serialization, frozen
`AutoModel` last-layer CLS features, and shared scorer shape: 128-hidden
`tanh` layer, dropout 0.1, scalar candidate logit, and soft-target cross
entropy over each question's 2–5 supplied options. The encoders, pooler, and
MLM/SPLADE vocabulary head were frozen or omitted. Only the readout was
trained. The two candidates were run separately with identical state splits,
seed 17, at most 8 epochs, patience 3, AdamW learning rate `2e-3`, weight
decay `1e-4`, train batch 128, and validation checkpoint selection by macro
soft NLL across workflow/question-type cells. Both selected epoch 8. No
temperature calibration or validation-driven prompt changes were made.

The full Japanese BERT checkpoint file contains unused pretraining weights;
the byte count is not the compact product export size. No export or ternary
quantization was performed. BERT has about 4.01 times the used encoder-plus-
readout parameters of the compact candidate.

## Fixed validation results

Accuracy is measured against each teacher distribution's modal option. NLL
and Brier use the full soft teacher distribution. ECE-15 is macro-averaged
within cells, comparing confidence with the teacher probability assigned to
the predicted option. The uniform-option baseline is expected accuracy over
modal labels and soft NLL/Brier, macro-averaged over the same 12 cells.

| Candidate | Macro modal Accuracy | Macro soft NLL | Macro soft Brier | Macro ECE-15 |
|---|---:|---:|---:|---:|
| Uniform over available options | 0.3271 | 1.1831 | 0.2554 | — |
| LightSPLADE-derived 27.6M | 0.4340 | 1.1300 | 0.2282 | 0.0588 |
| Japanese BERT-base 110.7M | 0.5340 | 1.0751 | 0.2010 | 0.0556 |

### By workflow and question type

| Cell | N | LightSPLADE Accuracy | BERT Accuracy | Delta | LightSPLADE NLL | BERT NLL |
|---|---:|---:|---:|---:|---:|---:|
| agent_trace_observability / choice | 120 | 0.375 | 0.408 | +0.033 | 1.326 | 1.305 |
| agent_trace_observability / noul | 60 | 0.433 | 0.550 | +0.117 | 0.702 | 0.693 |
| agent_trace_observability / score | 120 | 0.358 | 0.425 | +0.067 | 1.410 | 1.342 |
| customer_service / choice | 120 | 0.367 | 0.500 | +0.133 | 1.509 | 1.313 |
| customer_service / noul | 60 | 0.600 | 0.600 | +0.000 | 0.692 | 0.692 |
| customer_service / score | 120 | 0.208 | 0.433 | +0.225 | 1.346 | 1.224 |
| invoice_processing / choice | 60 | 0.233 | 0.467 | +0.233 | 1.324 | 1.300 |
| invoice_processing / noul | 120 | 0.700 | 0.742 | +0.042 | 0.602 | 0.555 |
| invoice_processing / score | 120 | 0.458 | 0.417 | -0.042 | 1.266 | 1.202 |
| security_incidents / choice | 60 | 0.683 | 0.683 | +0.000 | 1.260 | 1.251 |
| security_incidents / noul | 120 | 0.492 | 0.675 | +0.183 | 0.700 | 0.667 |
| security_incidents / score | 120 | 0.300 | 0.508 | +0.208 | 1.424 | 1.356 |

The macro accuracy gain is not uniform: BERT ties in two cells, loses in
`invoice_processing/score`, and gains in the other nine. The per-cell sample
counts are modest; several cells have 60 state groups.

### Paired uncertainty diagnostic

Saved predictions were matched by state ID and question key, then resampled
10,000 times with seed `20261123`, sampling 60 state groups with replacement
within each workflow. Every selected state carries all five of its questions
together. BERT minus LightSPLADE differences were:

| Metric | Difference | Percentile 95% interval | Resamples favoring BERT |
|---|---:|---:|---:|
| Macro modal Accuracy | +0.1000 | [+0.0660, +0.1340] | 100% |
| Macro soft NLL | -0.0549 | [-0.0668, -0.0422] | 100% |
| Macro soft Brier | -0.0273 | [-0.0340, -0.0205] | 100% |
| Macro ECE-15 | -0.0032 | [-0.0174, +0.0143] | 58% |

The bootstrap describes paired validation predictions. Both checkpoints were
selected using this same validation set, so it is not independent-test
inference and its intervals do not correct selection optimism. ECE does not
show a dependable improvement.

## Learning curves and cost

| Epoch | LightSPLADE train soft CE | Light validation macro NLL | Light validation Accuracy | BERT train soft CE | BERT validation macro NLL | BERT validation Accuracy |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.1942 | 1.1518 | 0.4313 | 1.1636 | 1.1144 | 0.4729 |
| 2 | 1.1750 | 1.1438 | 0.4368 | 1.1222 | 1.0951 | 0.4951 |
| 3 | 1.1700 | 1.1383 | 0.4417 | 1.1123 | 1.0859 | 0.5042 |
| 4 | 1.1658 | 1.1361 | 0.4389 | 1.1041 | 1.0825 | 0.5056 |
| 5 | 1.1617 | 1.1331 | 0.4361 | 1.1029 | 1.0861 | 0.5000 |
| 6 | 1.1607 | 1.1314 | 0.4396 | 1.0981 | 1.0774 | 0.5083 |
| 7 | 1.1572 | 1.1303 | 0.4424 | 1.0955 | 1.0828 | 0.5097 |
| 8 | 1.1566 | 1.1300 | 0.4340 | 1.0940 | 1.0751 | 0.5340 |

Train CE is a sample-weighted training quantity; validation NLL is macro over
12 workflow/question-type cells, so their numeric difference is not a fixed
example generalization gap. Validation NLL improved through epoch 8 for both
models; there is no clear NLL overfit signal within this short schedule, but
epoch 8 selection and only one seed remain limitations.

On the same local CPU environment (Python 3.11.9, PyTorch 2.6.0+cu124,
Transformers 5.6.2, four CPU threads), frozen feature extraction took 1,346.07
seconds for LightSPLADE and 5,949.86 seconds for BERT; readout optimization
took 1.96 and 2.40 seconds respectively. End-to-end wall time was 22m29s and
1h39m14s. BERT took 4.42x as long for these 21,300 candidate pairs. Neither
run used the GPU or paid compute.

The tokenized pair length ranged to 877 tokens, with mean 433.65. At the
shared 512-token cap, 6,638 / 21,300 candidate pairs (31.16%) were truncated.
Truncation was `only_first`: candidate/question text was preserved and the
tail of the state segment could be removed. No candidate prompt alone exceeded
512 tokens. This is a material limitation for long states and should be
addressed only in a separately specified comparison.

## Runtime artifact locations and hashes

Artifacts are external to Git under
`C:\CodexArtifacts\pretrained-reuse-r2-local\`:

- LightSPLADE full run: `typed-decisions-ja-light-splade-full-fixed-20261123-attempt-02/light-splade/`
- BERT full run: `typed-decisions-ja-bert-base-full-fixed-20261123-attempt-01/bert-base/`
- Paired bootstrap: `typed-decisions-ja-bert-vs-light-splade-paired-bootstrap-20261123-attempt-01/paired-bootstrap.json`
- Combined machine-readable run manifest: `typed-decisions-ja-encoder-comparison-manifest-20261123-attempt-01/comparison-manifest.json`
- Safely stopped memory-bug attempt: `typed-decisions-ja-light-splade-full-20261123-attempt-01/attempt-status.json` (no checkpoint or features)

| Artifact | SHA-256 |
|---|---|
| LightSPLADE `run-report.json` | `53e0d426318d3a7f26e8b78b2b92820035dc5a697b5e9b054834100c001177d9` |
| LightSPLADE `best-readout.pt` | `e10043e9b12682a60440387b0a597a2084f856849366eedb59320b23bc3e62e5` |
| LightSPLADE validation predictions | `07d7a4b038363e85bb4a86a2f723937f8fa05b4701b46fb285b4481de0a89c0f` |
| BERT `run-report.json` | `70311de474491ec7f5c12a3e2bab2edf245470ce69290f9351993d9aa75cae42` |
| BERT `best-readout.pt` | `962c8e2f61694f38d9e55ae3befba4d44624009e7e1a3917d14adc505fabed8e` |
| BERT validation predictions | `e207469938541c2e259721c565a7381869ac6d13798c98e5f3e34b81e3795f09` |
| Paired bootstrap JSON | `78816392655d4bd2378386a853e70cf86d28c0562832c46575b58f7b14280a0d` |
| Full comparison script | `b224a0bb8618e0dc69d9c0cec788bfead0699eeab2a4d668411c5574482c58b5` |
| Bootstrap analysis script | `e9e655ef4cb099b2a954e05efa6494c42c58b287fb36cf4f3ac5d414c1527a74` |
| Combined run manifest | `c883ad6e13d93ac672cb31278b8061b3ba26af64c96219aa683754872c750d0e` |

The executed comparison used the CLI below for each model, changing only the
encoder name/path, `--encode-batch-size`, and output directory. Model-specific
encode batch sizes were 8 for LightSPLADE and 4 for BERT-base; both preserve
the same candidate rows, order, truncation rule, and final feature dimension.

```powershell
$env:PYTHONPATH = 'src;C:\CodexArtifacts\pretrained-reuse-r2-local\light-splade-ja28m-runtime-audit-20261123-attempt-01\deps'
$env:CUDA_VISIBLE_DEVICES = ''
python scripts/compare_typed_decisions_ja_encoders.py `
  --train-parquet 'C:\CodexArtifacts\pretrained-reuse-r2-local\typed-decisions-ja-train-only-screen-20261123-attempt-01\all\train-00000-of-00001.parquet' `
  --split-assignments 'C:\CodexArtifacts\pretrained-reuse-r2-local\typed-decisions-ja-train-only-screen-20261123-attempt-01\frozen-train-only-split-assignments.jsonl' `
  --dataset-revision 01a4558e83bf5a748094239df843d658b33d2533 `
  --tokenizer-dir 'C:\CodexArtifacts\pretrained-reuse-r2-local\jnli-tohoku-bertbasev3-model-20261123-attempt-01' `
  --encoder light-splade='C:\CodexArtifacts\pretrained-reuse-r2-local\light-splade-ja28m-runtime-audit-20261123-attempt-01' `
  --output-dir 'C:\CodexArtifacts\pretrained-reuse-r2-local\typed-decisions-ja-light-splade-full-fixed-20261123-attempt-02' `
  --seed 17 --epochs 8 --patience 3 --batch-size 128 --encode-batch-size 8 --max-length 512
```

For the BERT-base invocation, replace `light-splade` with `bert-base`, set the
encoder path to `jnli-tohoku-bertbasev3-model-20261123-attempt-01`, use
`--encode-batch-size 4`, and output to
`typed-decisions-ja-bert-base-full-fixed-20261123-attempt-01`. The paired
bootstrap was run with 10,000 repetitions and seed `20261123` using
`scripts/analyze_typed_decisions_ja_comparison.py`.

## Validation performed

On the isolated R2 worktree, full `ruff check .` passed, full `pytest -q`
passed (**2,903 passed, 11 skipped**), and the CI workflow's model-manifest,
dataset-manifest, catalog, and example-manifest audit commands all passed.
Both selected readouts reload with a maximum probability delta of exactly
zero. Model weights, feature caches, source media, and full validation
predictions remain external to Git.

The run reports also contain full prediction records, epoch histories, model
weight hashes, feature-cache hashes, environment versions, input hashes, and
exact reloaded-readout probability checks. The R2 worktree base at launch was
`5cecdc7b2288f280fc0c13f240e0b673120de6c0`; the scripts and tests were
uncommitted at execution and are pinned above by SHA-256.

## Decision and next step

Keep Japanese BERT-base as a strong text teacher/reference, but its one-tower
size and 4.42x CPU feature cost make it a poor default replacement in a
four-modality model. The compact retrieval-derived encoder is smaller but
weaker in this CLS-readout probe. This does not establish that the SPLADE
architecture itself is poor: its trained tokenwise vocabulary head and sparse
pooling were intentionally omitted, so this specifically evaluates a frozen
CLS representation with a shared Decision scorer.

The next text experiment should test a pinned compact Japanese BERT-family
encoder trained or distilled for sentence/option representation under a
newly admitted source, keeping the BERT-base teacher and current Ruri path as
references. It must use the full quantized-resident parameter accounting and
should not revisit these exposed validation IDs for repeated tuning. The
source's translation/teacher-label limitations mean that any better score
still needs confirmation on independently labeled decision data. The broader
R2 multimodal target remains open.

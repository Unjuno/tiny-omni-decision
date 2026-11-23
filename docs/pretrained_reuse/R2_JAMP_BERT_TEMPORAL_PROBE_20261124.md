# R2 Japanese BERT Temporal Decision Probe — 2026-11-24

## Question and scope

Does a Japanese BERT-family representation improve a compact text Decision readout on temporal natural-language inference, and does upper-layer adaptation help probability quality? This is a task-scoped text probe, not evidence of multimodal quality or a general-purpose Decision model. The source is the public Jamp temporalNLI dataset at pinned revision `f578b5daac7bff3855d238c032a60fd047cddd9c`. Its repository declares CC BY-SA 4.0; the paper describes template generation and notes that content words came from a separately distributed GSK case-frame dictionary. Only the public Jamp release was used; no GSK source dictionary was obtained.

Sources: [Jamp dataset](https://github.com/tomo-vv/temporalNLI_dataset), [ACL 2023 paper](https://aclanthology.org/2023.acl-srw.8/), [GSK distribution conditions](https://www.gsk.or.jp/catalog/gsk2018-b).

## Data protocol

Only official `train_all.tsv` was used to make a deterministic seed-17 grouped split by `template_num`. Five folds were balanced by label and row count; fold 0 is validation. The resulting frozen split has 7,960 train rows in 62 template groups and 1,990 validation rows in 17 groups. There is no template, source ID, or normalized premise/hypothesis overlap across train and validation. The source TSV SHA-256 is `711f496ffd7806b83e13502af6b91774f8e05f1fb13215bd33c6824607e3b07b`; train, validation, and split manifest hashes are in `manifests/pretrained_reuse/jamp-bert-temporal-probe-20261124.json`.

The official `dataset/test_2.tsv` was not read during candidate fitting. After the selection protocol was committed, a preflight parsed the test rows and found `num` collisions with train and validation. The `num` field is file-local, so cross-file collisions are recorded but are not treated as source identity. Normalized premise/hypothesis overlap is the content identity gate and was zero. The preflight computed no model predictions or test metrics. It also found that test `template_num` values overlap all 62 train and 17 validation template groups. Therefore the official test can compare held-out records within known template families, but cannot measure unseen-template generalization. The one-shot evaluator uses validation-selected checkpoints only, saves per-row predictions and calibration metrics, and refuses to overwrite an evaluation output. The cross-candidate rule is frozen to the lowest validation NLL; that is the frozen four-layer LINE DistilBERT readout. The publisher test is a public task test, not a sealed audit. Its result will not be used for tuning.

## Matched frozen-encoder comparison

All candidates use the same mean pooling over the premise/hypothesis pair, 128-unit tanh readout, three logits, seed 17, and 30 fixed epochs. Each candidate receives 7,960 train and 1,990 validation rows. Best checkpoint selection uses validation NLL with a 0.001 minimum improvement. The RTX 3080 Laptop run used about 601 MB peak allocated VRAM and completed in about 45 seconds total.

| Encoder | Encoder + readout parameters | Four-tower estimate replacing Ruri | Best epoch | Validation accuracy | Macro class accuracy | NLL | Brier | ECE-15 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Ruri-v3-30m | 36,738,819 | 178,722,435 | 1 | 38.69% | 37.16% | 1.1465 | 0.6855 | 0.0841 |
| LINE Japanese DistilBERT, first 4 layers | 54,010,883 | 195,994,499 | 1 | 45.83% | 44.64% | 1.0950 | 0.6530 | 0.0754 |
| Tohoku Japanese BERT-base v3 | 111,305,987 | 253,289,603 | 2 | 47.64% | 47.16% | 1.1011 | 0.6520 | 0.0850 |

The BERT-base model has slightly higher accuracy, but worse validation NLL than the in-budget DistilBERT candidate and exceeds the provisional four-tower parameter budget. It is useful as a text reference, not a current product candidate. All three curves show overfitting: train metrics continue rising while validation NLL and calibration deteriorate. For example, DistilBERT validation NLL rises from 1.095 at epoch 1 to 1.631 at epoch 30.

**Provenance limitation:** the frozen comparison runner was not committed when it ran. Its SHA-256 (`f1e08c11287a3e17ee7e12ff4fbb046c6922800d4b16ca086dff0115dee74dae`) and effective config hash (`5995bdaa36fee530e4abb5051988afb4542643d441bb81d95bc57ab03cfe476a`) were recorded in the run output; the runner has since been committed as part of this branch. Do not describe the run as having executed from its recorded source commit.

## Upper-layer adaptation

An initial exploratory run accidentally loaded all six DistilBERT layers rather than the intended four. It is retained separately and excluded from the in-budget comparison: its projected four-tower estimate was 210.17M parameters. The corrected run loads the first four pretrained layers and updates only layers 2–3 plus the readout; embeddings and layers 0–1 remained frozen, verified by unchanged frozen-weight hash.

The corrected run used seed 17, 12 fixed epochs, 95,520 total row exposures, and selected epoch 1 by validation NLL. The selected checkpoint scored 45.63% accuracy, 1.4513 NLL, 0.7218 Brier, and 0.1841 ECE-15. At epoch 12 accuracy rose to 69.80%, but NLL worsened to 2.6063, Brier to 0.5438, ECE to 0.2357, and mean confidence to 0.9337. Thus the accuracy increase came with severe overconfidence and worse probability quality. This run does not support adopting the tested fine-tuning recipe.

## Decision and next step

The evidence supports keeping compact Japanese BERT as a text representation candidate and reference, while rejecting the current upper-layer fine-tuning schedule for calibrated decisions. The frozen four-layer DistilBERT is the predeclared baseline for the one-time publisher test comparison, alongside the other validation-selected candidates. This is a single synthetic temporal-text task; it cannot establish the full Omni objective, 95–98% performance, or multimodal fusion. No sealed audit, modality data, or product checkpoint was used or promoted.

## One-shot official publisher-test result

The committed validation-only rule chose `line-distilbert-ja-first4` as the fixed baseline (NLL 1.09498). A single local CUDA evaluation then scored four preselected checkpoints on all 348 `test_2.tsv` rows. The test label counts are contradiction 116, entailment 114, and neutral 118. No candidate was selected or changed using these results.

| Candidate | Validation selected Accuracy / NLL | Publisher-test Accuracy | Macro class Accuracy | NLL | Brier | ECE-15 | Mean confidence | Four-tower estimate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Ruri-v3-30m frozen | 38.69% / 1.1465 | 34.77% | 34.55% | 1.1018 | 0.6699 | 0.1172 | 0.4636 | 178.72M |
| LINE DistilBERT first 4 layers frozen | 45.83% / 1.0950 | 47.41% | 47.25% | 1.0192 | 0.6175 | 0.1129 | 0.4940 | 195.99M |
| Tohoku BERT-base v3 frozen | 47.64% / 1.1011 | 56.32% | 56.29% | 0.9522 | 0.5738 | 0.0739 | 0.5233 | 253.29M |
| LINE DistilBERT first 4 layers, top 2 adapted | 45.63% / 1.4513 | 56.90% | 57.17% | 0.8956 | 0.5441 | 0.0936 | 0.6491 | 195.99M |

The paired 10,000-replicate bootstrap resampled the 79 test `template_num` groups (seed 1729). Relative to frozen DistilBERT, frozen BERT-base had +8.91 percentage points of Accuracy (95% interval +2.20 to +15.78), -0.0675 NLL (-0.1182 to -0.0189), and -0.0441 Brier (-0.0792 to -0.0108). The adapted DistilBERT had +9.42 points Accuracy (+1.90 to +17.30), -0.1223 NLL (-0.2357 to +0.0046), and -0.0728 Brier (-0.1403 to -0.0048). These are descriptive intervals conditional on a test whose template families were present in development; they do not provide an unseen-template estimate.

This result strengthens the evidence that Japanese BERT representations help this narrow temporal text task. BERT-base exceeds the provisional four-tower parameter ceiling. The in-budget adapted DistilBERT had the strongest observed test scores, but its validation NLL was substantially worse than the frozen baseline and this test may not be used to promote it. Keep the frozen DistilBERT as the validation-selected reference; do not tune against this exposed test. All accuracy values remain far below 95–98%.

The official test `num` field collides with train/validation IDs (338 and 10 rows) because it is file-local. Normalized premise/hypothesis overlap was zero and there were no duplicate normalized pairs within the test. All 62 train and 17 validation template IDs also appear in the test. The test is therefore record-held-out but not template-held-out.

The test file SHA-256 is `10001a07e00b8a21f370838f6cd3869c7edfb1fd2b7b69cfcefebea68009bf07`; its pinned Git blob is `de331fa36cbddfddbb178d7e3edda90139851a7a`. Result JSON SHA-256 is `7f18673336ba47d3ead10c5b379ce46dfd5689314b50154348c1e198b88ab421`. The run used RTX 3080 Laptop, CUDA 12.4, torch 2.6.0+cu124, and no paid compute. Elapsed time was about 14 seconds from the command lifecycle; per-run wall time and peak evaluation VRAM were not serialized by this evaluator. Selected checkpoint hashes and per-row predictions are in the external result JSON and companion JSONL files under `C:\CodexArtifacts\pretrained-reuse-r2-local\jamp-bert-temporal-probe-20261124-attempt-01\publisher-test-once\evaluation-02\`.

## Artifacts

Run outputs and model files are external to Git under `C:\CodexArtifacts\pretrained-reuse-r2-local\jamp-bert-temporal-probe-20261124-attempt-01\`. They include the frozen comparison report, effective config, curves and predictions, plus both separate fine-tuning attempts. The code and split builder are in `scripts/pretrained_reuse/`; tests are in `tests/test_build_jamp_grouped_split.py` and `tests/test_jamp_distilbert_freezing.py`.

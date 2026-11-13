# Completed lexical-range training control

Execution PASS:2,048updates, selected1408. Source5793e6c. Exact saved-best reload and frozen-parameter hash checks passed. All pinned inputs and best/final hashes rechecked after completion. No legacy artifact changed. Total22,713,601parameters;10,647,169trainable. New optimizer warm start, not strict resume.

| Step | Original Acc | Original NLL | Curriculum Acc | Curriculum NLL | Selector |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.60236 | 0.87743 | 0.80566 | 0.46110 | 0.66926 |
| 128 | 0.61220 | 0.91042 | 0.78955 | 0.50536 | 0.70789 |
| 256 | 0.61575 | 0.90287 | 0.78320 | 0.52402 | 0.71344 |
| 384 | 0.62165 | 0.89133 | 0.77588 | 0.53320 | 0.71227 |
| 512 | 0.60197 | 0.93486 | 0.79248 | 0.52991 | 0.73238 |
| 640 | 0.60236 | 0.94141 | 0.80127 | 0.53768 | 0.73954 |
| 768 | 0.60669 | 0.94960 | 0.80664 | 0.50732 | 0.72846 |
| 896 | 0.59685 | 1.03321 | 0.79883 | 0.48500 | 0.75911 |
| 1024 | 0.59409 | 1.03202 | 0.82227 | 0.46824 | 0.75013 |
| 1152 | 0.59843 | 1.03576 | 0.80713 | 0.44731 | 0.74153 |
| 1280 | 0.60472 | 0.92749 | 0.82373 | 0.44631 | 0.68690 |
| 1408 | 0.60433 | 0.90770 | 0.82959 | 0.42328 | 0.66549 |
| 1536 | 0.60354 | 0.92147 | 0.79883 | 0.49641 | 0.70894 |
| 1664 | 0.61260 | 0.90280 | 0.82373 | 0.43784 | 0.67032 |
| 1792 | 0.59016 | 0.99008 | 0.82275 | 0.44610 | 0.71809 |
| 1920 | 0.60591 | 1.01324 | 0.82959 | 0.44945 | 0.73134 |
| 2048 | 0.60591 | 0.94197 | 0.83008 | 0.43738 | 0.68968 |

| Endpoint / corpus | Accuracy | NLL | Brier | ECE |
| --- | ---: | ---: | ---: | ---: |
| selected / original | 0.60433 | 0.90770 | 0.50994 | 0.08021 |
| selected / curriculum | 0.82959 | 0.42328 | 0.23094 | 0.02341 |
| final / original | 0.60591 | 0.94197 | 0.50874 | 0.07046 |
| final / curriculum | 0.83008 | 0.43738 | 0.23813 | 0.04660 |

Whole stage:16,384unique consumed question IDs, 6056state IDs. Selected1408:11,264unique consumed question IDs, 5039state IDs. Per-source/task counts are in the full history. Lifetime unique coverage is smaller than summed stage counts because prior training is inherited and IDs repeat.

Controlled config, initial weights, sample ID order and validation IDs/order match the numeric-only warm-start run. Full8,192-row comparison confirms changes only to question text and wording metadata; numeric values, states, options and labels match. All four wording variants contain2,048questions; their joint numeric-scale counts are recorded rather than assumed exactly balanced.

The selected narrow-curriculum result alone cannot demonstrate wording generalization. The fixed selector never loaded paraphrase transfer data. No new transfer inference has occurred for this model, and it is not promoted. Final-versus-selected regressions are preserved; do not call completion or a falling rolling training loss proof of no overfitting.

Local training/evaluation/reload elapsed 1123.78s excluding initial load/tokenization, peak CUDA allocation 3473190400bytes, no cloud charges. Artifact directory `C:\CodexArtifacts\pretrained-reuse-r2\conditional-repair-lexical-range-01`. Exact argv, UTC start, environment-independent input hashes and all per-task metrics in JSON. Best SHA `fccee21e0e5ffbaffdee899a65fe6538f52f3b788752a56b6b5b6388cad15abb`, final SHA `cba9e188539612aa8d886a53a70bff766de9ecae440199222d73472f7c11f824`. Both weights-only, no exact resume.

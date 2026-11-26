# Comparison-vocabulary training result

Execution PASS: 2,048 updates; earliest minimum validation mean NLL selects step 1,664. Source 505504f. Exact saved-best reload and frozen-parameter checks passed. This is a development experiment, not general-quality completion.

| Step | Original Acc | Original NLL | Curriculum Acc | Curriculum NLL | Selector |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.60236 | 0.87743 | 0.80566 | 0.46110 | 0.66926 |
| 128 | 0.61575 | 0.90933 | 0.79004 | 0.51641 | 0.71287 |
| 256 | 0.61496 | 0.90400 | 0.78027 | 0.51773 | 0.71086 |
| 384 | 0.62165 | 0.89591 | 0.78320 | 0.53306 | 0.71449 |
| 512 | 0.60236 | 0.92913 | 0.78125 | 0.55281 | 0.74097 |
| 640 | 0.60079 | 0.96417 | 0.79004 | 0.56063 | 0.76240 |
| 768 | 0.60748 | 0.93332 | 0.79102 | 0.51974 | 0.72653 |
| 896 | 0.60197 | 1.03409 | 0.79443 | 0.50139 | 0.76774 |
| 1024 | 0.59409 | 1.04037 | 0.81689 | 0.46181 | 0.75109 |
| 1152 | 0.60157 | 1.04493 | 0.80371 | 0.46520 | 0.75507 |
| 1280 | 0.60866 | 0.93741 | 0.81396 | 0.43550 | 0.68645 |
| 1408 | 0.60669 | 0.90723 | 0.82227 | 0.43215 | 0.66969 |
| 1536 | 0.60354 | 0.91588 | 0.80225 | 0.46838 | 0.69213 |
| 1664 | 0.61535 | 0.88524 | 0.81543 | 0.44916 | 0.66720 |
| 1792 | 0.59685 | 0.96536 | 0.82080 | 0.44906 | 0.70721 |
| 1920 | 0.60472 | 1.01459 | 0.82422 | 0.42406 | 0.71933 |
| 2048 | 0.60709 | 0.95003 | 0.82471 | 0.45694 | 0.70349 |

| Endpoint / corpus | Accuracy | NLL | Brier | ECE |
| --- | ---: | ---: | ---: | ---: |
| selected / original | 0.61535 | 0.88524 | 0.49500 | 0.06748 |
| selected / curriculum | 0.81543 | 0.44916 | 0.25232 | 0.02751 |
| final / original | 0.60709 | 0.95003 | 0.51061 | 0.08338 |
| final / curriculum | 0.82471 | 0.45694 | 0.24587 | 0.04607 |

Whole stage: 16,384 consumed / 16,384 unique question IDs / 6,056 states. Selected checkpoint: 13,312 consumed / 13,312 unique questions / 5,507 states. These are stage counts, not lifetime unique exposure.

Same 22,713,601 resident parameters, 10,647,169 trainable. New optimizer from encoder-coverage step2048, not exact resume. Seed17, fixed 2,048 updates, existing selector and validation IDs/order preserved. Only comparative vocabulary differs from the lexical-range training control. No validation-conditioned data changes.

The selected selector improves only slightly versus initialization; final selector regresses. Preserve this retention/calibration trade-off. Rolling train CE is not a fixed-data generalization gap. Original-text quality and general multimodal 95% remain unmet.

Local RTX3080 Laptop; 1121.47s excluding initial load/tokenization; peak CUDA allocation 3473190400 bytes; cloud cost $0. Training source 505504f. All effective arguments, input hashes, task counts, evaluation timings and gradients are in the companion JSON.

Artifacts: `C:\CodexArtifacts\pretrained-reuse-r2\conditional-repair-comparison-vocabulary-01`. Best SHA `868cc435f3b70ce84d0c49781d80680ba28a1078cbcf6de25f53a58756661b13`; final SHA `cb4baa2906b22cfcf518bf9724eed9255935a35652c343b893ae4d0295da5c50`. Existing models and artifacts unchanged. Transfer results are reported separately; training validation alone is not wording-transfer evidence.

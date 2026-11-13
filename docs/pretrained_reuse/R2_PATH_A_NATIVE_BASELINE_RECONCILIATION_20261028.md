# R2 Path A native Decision baseline reconciliation (2026-10-28)

Status: **the planned frozen-feature Path A baseline and its LSTM comparison
were already trained and replayed.** Do not repeat them. Their scores are a
development baseline, not a product-quality result or blind audit.

## Run and integrity

The source run is `72c1eb47dde0c81e503d3a4d783aa5528660186e` with fixed config
SHA-256
`45954a7240ee8da32e4b5b5dc3283994c40c4849451f27ff4fce13d3cc118f75`.
Seed 17 and 2,048 optimizer updates were used for both arms. The four frozen
feature encoders were not updated. Both arms used the same initialization,
feature pins, corpus, sampler, option-order rule, validation IDs and selector;
initial validation logits were exactly equal. The selector was hierarchical
task/source/modality macro NLL with `min_delta=1e-5`, earliest step on ties.
Best and final checkpoints were retained separately and fresh-process replay
was exact. Sealed audit and cloud compute were not used.

| Arm | Trainable params | Total pre-quantization params | Best step | Main run wall time | Peak CUDA allocated |
| --- | ---: | ---: | ---: | ---: | ---: |
| Path A baseline | 889,345 | 196,254,874 | 1,792 | 562.97 s | 208,541,696 B |
| Baseline + LSTM | 1,925,761 | 197,291,290 | 1,536 | 1,067.02 s | 275,454,464 B |

Each full run consumed 8,192 presentations, 6,656 unique questions, 1,536
repeats, 5,107 unique observations, 3,114 media assets, and 3,382 groups. At
the selected checkpoint, baseline had consumed 7,168 presentations / 5,888
unique questions; LSTM had consumed 6,144 / 5,120. The validation set contains
24,346 development questions. These are not blind independent test metrics.

External output directories:

- `C:/CodexArtifacts/pretrained-reuse-r2/native-common-baseline-fit-01/`
- `C:/CodexArtifacts/pretrained-reuse-r2/native-common-lstm-fit-01/`

Receipt SHA-256: baseline result
`aaa1fe96a4be85a71ce146981c6b2e8b58e1308d91fc110e1ece6804b4d8048c`; LSTM
result `47dde479c91ed82cdbf260686f0184ebdf95b53981b28248e0ad1ebf2c6b10c6`.
Best checkpoint SHA-256: baseline
`3eed6558731de60da82e75e38e599699f29663dd601f942505fec26a48b8dc1d`; LSTM
`72c71f853223b7a9be7d3039504b81131e0938f81831d5bcefe7db6c8016205a`.

## Fixed 2,048-update comparison

Scores use the predeclared task/source/modality macro aggregation, not the
sample-weighted overall metrics in the raw receipts.

| Arm | Modality | Accuracy | NLL | Brier | ECE |
| --- | --- | ---: | ---: | ---: | ---: |
| Baseline | Text | 18.07% | 2.0720 | 0.8486 | 0.0952 |
| Baseline | Image | 19.73% | 2.1623 | 0.8646 | 0.0608 |
| Baseline | Audio | 53.03% | 1.1854 | 0.5494 | 0.0909 |
| Baseline | Video | 31.50% | 1.3444 | 0.7035 | 0.0920 |
| Baseline | Macro | 30.58% | 1.6910 | 0.7415 | 0.0847 |
| LSTM | Text | 18.59% | 2.1004 | 0.8498 | 0.0938 |
| LSTM | Image | 17.82% | 2.2004 | 0.8744 | 0.0598 |
| LSTM | Audio | 53.17% | 1.1900 | 0.5537 | 0.0970 |
| LSTM | Video | 33.25% | 1.3375 | 0.6995 | 0.0609 |
| LSTM | Macro | 30.71% | 1.7071 | 0.7444 | 0.0779 |

The best-validation checkpoints do not reverse the decision:

| Arm | Selected step | Macro Acc | Macro NLL | Video Acc | Minimum task Acc |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 1,792 | 30.71% | 1.6865 | 31.50% | 0% |
| LSTM | 1,536 | 30.19% | 1.6967 | 35.75% | 0% |

LSTM's selected Video score comes with lower Image accuracy and lower macro
accuracy. Its 2,048-step Video gain is small, while macro NLL worsens. A
scene-cluster paired bootstrap on the 200 development scenes found a fixed-end
Video Accuracy difference of +1.75 percentage points with 95% interval
[-1.50, +5.00]; temporal-descriptive Accuracy difference was 0.00 points with
95% interval [-3.00, +3.00]. The comparison does not establish a recurrent
memory benefit. The selected minimum task Accuracy of zero comes from the
STARSS event setup's positive-class recall failure; aggregate Accuracy must
not be treated as success.

## Decision and remaining work

Keep both arms as historical research baselines; do not adopt the LSTM, tune
these exposed validation IDs, or infer a general capacity ceiling. This run
does complete the planned Path A baseline-training step, but does not satisfy
the objective: all four modality/task suites remain far below the 95–98%
aspiration, the overall task minimum is zero, cross-modal benefit is unproven,
and joint/OOD performance, independent audit, deployment package, and
end-to-end latency remain open.

The original full report, learning curves, task/source metrics, bootstrap,
environment, artifact hashes and test/CI verification are on the source
branch at [`R2_NATIVE_TEMPORAL_FIT_RESULTS_20261016.md`](https://github.com/Unjuno/tiny-omni-decision/blob/87ed587ee847b83531b6be186f8694685124b731/docs/pretrained_reuse/R2_NATIVE_TEMPORAL_FIT_RESULTS_20261016.md).
Full-extent, coverage and long-budget follow-ups remain in their separate
reports; this reconciliation does not replace or modify them.

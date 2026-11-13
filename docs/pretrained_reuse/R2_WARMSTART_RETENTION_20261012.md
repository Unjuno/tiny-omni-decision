# Saved-prediction retention audit

No inference or training. Same completed warm-start run, step0 versus selected1408 and final2048. Verify frozen corpus hash, complete history, earliest minimum selector, unique ordered IDs, options, targets and finite normalized probabilities before pairing. Resample256states,10,000replicates, seed20261012; percentile95% intervals. These posthoc development intervals do not correct for selection or multiple comparisons.

| Endpoint | Task | Accuracy difference (pp) | 95% interval (pp) |
| --- | --- | ---: | --- |
| selected | overall | 1.66 | [0.00, 3.32] |
| selected | conditional_selection | 6.25 | [1.17, 11.33] |
| selected | count | 6.64 | [-1.17, 14.06] |
| selected | extremum | 0.20 | [-1.76, 2.15] |
| selected | temporal_order | -0.20 | [-2.34, 2.15] |
| final | overall | 3.22 | [1.46, 4.98] |
| final | conditional_selection | 9.77 | [3.91, 15.62] |
| final | count | 14.45 | [7.03, 21.88] |
| final | extremum | 0.59 | [-1.37, 2.54] |
| final | temporal_order | -0.39 | [-2.73, 1.95] |

The selected overall interval includes zero. Conditional selection improves descriptively; count is uncertain at the selected checkpoint. Final improvements concentrate in conditional selection/count, while extrema and scalar time ordering remain essentially unchanged. Scalar time ordering is not video reasoning. These results do not justify selecting the final checkpoint instead of the predefined NLL-selected checkpoint.

| Original text task | Initial NLL | Selected NLL | Final NLL | Initial ECE | Selected ECE | Final ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| choice | 1.02235 | 1.05061 | 1.12782 | 0.03322 | 0.05207 | 0.08376 |
| noul | 0.54844 | 0.55779 | 0.58056 | 0.02307 | 0.05876 | 0.07537 |
| score | 1.20521 | 1.24111 | 1.29393 | 0.04347 | 0.09931 | 0.10771 |

Calibration/retention losses must remain visible even where Accuracy improves. Training fit is not held-out transfer. The next fixed-weight comparison uses the selected checkpoint only, with new state identities and separately varied ranges/wording. No training-budget extension or repeated old-transfer tuning.

Reproduce: `python -m scripts.audit_warmstart_retention --root C:/CodexArtifacts/pretrained-reuse-r2 --output <new-file.json>`. Source, prediction and corpus hashes are retained in the adjacent JSON. Existing artifacts unchanged.

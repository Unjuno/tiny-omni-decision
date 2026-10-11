# Approved fixed conditional-ranking diagnostic

User approved 64 states x four conditions (price/risk, lowest/highest), seed17,
three candidate IDs, fixed adapted epoch-3 checkpoint, inference only. Protocol
and generated inputs were saved before inference; source commit3b5a8ad.
Prices are distinct integers10..999 and risks distinct integers1..99; candidate
order is A/B/C. All generated states are retained. Oracle tests independently
sort the values and verify opposite conditions have distinct correct answers.
This is a constructed diagnostic, not a replacement benchmark or blind audit.

## Result

Execution PASS; conditional-ranking usefulness NOT DEMONSTRATED.
256questions: Accuracy35.94%, NLL1.15890, Brier0.70772, ECE0.15749.
Uniform three-way reference: expected Accuracy33.33%, NLL1.09861, Brier0.66667.
Target counts A78/B92/C86. **The model selected B for all256questions**,
so Accuracy equals the B frequency. Prediction flips:0/128 opposite-condition
pairs; both conditions correct:0/128. All questions fit (maximum89tokens),
so truncation cannot explain this result.

By condition: lowest price31.25%, highest price35.94%, lowest risk31.25%,
highest risk45.31%. These differences arise from fixed-label frequency and
must not be interpreted as comparative condition competence.

This contradicts interpreting the earlier synthetic validation gain as general
conditional numerical reasoning. State reliance on the original corpus did not
establish instruction following here. Candidate-label priors, serialization,
pretraining/task mismatch and numerical reasoning are possible explanations;
this single template does not isolate them. No label rotation, template search,
new training, recalibration or checkpoint selection was performed after results.
Do not promote this scorer as a general Decision runtime based on58.11% alone.

RTX3080 Laptop, FP32, PyTorch2.6.0+cu124, batch32candidate pairs, local pinned
model;1.51sec timed load/inference, peak allocated147,825,664bytes. No cloud cost.
Weights/data from previous runs unchanged; no sealed data opened. Adjacent JSON
records revision, hashes, full metrics and output paths. Predictions and inputs
are retained in the external artifact directory.

The next architectural decision must address conditional interpretation, while
preserving the100–200M total pre-quantization parameter target. Simply integrating
this text scorer or adding more same-corpus epochs is not justified by this result.

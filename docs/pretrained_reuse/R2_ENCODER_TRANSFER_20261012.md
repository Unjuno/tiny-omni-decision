# Fixed-format transfer diagnostic after encoder adaptation

Source187a088. Execute the preserved256-question/64-state diagnostic once with
selected step2048 from the encoder-coverage experiment. Same bare candidates,
question/state rendering, temperature1, single classifier score per candidate.
No additional training, calibration, prompt changes, checkpoint selection or
sealed audit. Immutable input SHA-256
8737946632cf65dd85b6360e9ed1f7b7b70ed7b59df230a0a7f5c9698233fb74
matches the original diagnostic exactly. New checkpoint is explicitly pinned by
SHA and its exact six-block/classifier parameter names; legacy default remains
pinned to its original checkpoint. No loose latest-checkpoint loading.

## Measured result

| Metric | Original adapted model | Encoder-coverage model |
| --- | ---: | ---: |
| Accuracy | 0.35938 | 0.33594 |
| NLL | 1.15890 | 1.42774 |
| Brier | 0.70772 | 0.81407 |
| ECE15 | 0.15749 | 0.26841 |
| Changed condition pairs /128 | 0 | 67 |
| Both-correct condition pairs /128 | 0 | 10 |

All256 inputs fit without truncation (max89tokens). Price lowest/highest each
29.69%, risk lowest17.19%, risk highest57.81%. Local FP32 GPU execution1.530sec,
peak PyTorch allocation147,825,664bytes; no cloud cost. Adjacent JSON records
all per-group calibration metrics, exact model/input hashes and artifact paths.

## Interpretation

Execution passes, useful transfer does not. The80.57% same-generator curriculum
result must not be presented as general conditional reasoning. This diagnostic
jointly changes candidate identifiers/count, numeric ranges, field names,
state schema and question wording relative to the curriculum. It cannot identify
which factor dominates. Answer changes alone are insufficient: only7.81% of
opposite-condition pairs are both correct, and confidence quality worsens.
This is a known development failure, not a blind test or a model-family ceiling.

Do not search wording or select a checkpoint on these256 questions. Preserve the
failed result. Next data work should define broad independent schema/lexical/
cardinality generalization splits before any run, retaining original-task
validation and the successful same-generator reference. A new recipe must be
judged on that predeclared coverage, not repeated optimization of this probe.
No claim of four-modality95% readiness, product promotion or compression follows.

Command: `python -m scripts.evaluate_text_condition_probe --model C:/CodexArtifacts/pretrained-reuse-r1/crossencoder-minilm-l6-audit-01/model --adaptation C:/CodexArtifacts/pretrained-reuse-r2/conditional-repair-encoder-coverage-01 --output C:/CodexArtifacts/pretrained-reuse-r2/encoder-condition-transfer-01 --checkpoint-reference encoder_coverage --candidate-style bare`.

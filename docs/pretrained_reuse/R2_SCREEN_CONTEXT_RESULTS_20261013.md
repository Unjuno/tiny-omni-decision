# Screen full-context comparison: rejected

Fixed30epoch CPU runs completed with identical initial tensors, ordered batches,
1536-input Linear8 heads (12,296parameters each), same1017train/723validation
regions and app split. Frozen22,056,576-parameter DINOv2 features only. Train-only
normalization; AdamW .001/.01,seed17,batch64. Full histories and every epoch's
predictions preserved. Earliest minimum validation NLL includesepoch0.

| Arm | Selected epoch | Accuracy | NLL |
|---|---|---|---|
| crop + zero |1|.323651|2.083574|
| crop + full-screen |0|.130014|2.312196|

Adding full-screen features did not help this model/task. The selected context
model is untrained; its shuffled-context comparison is consequently not evidence
of learned semantic reliance. Do not adopt, add epochs or claim context is
universally useless. Whole-screen224x224 pooling may erase details; additive
linear context cannot express arbitrary region-context interactions. These are
limitations, not measured causes. This is oracle-region classification with four
observed validation apps, not complete screen action selection.

Best and final weights separate; exact selected reload in both arms. Results JSON
contains per-class/app metrics, Brier/ECE/balanced Accuracy, shifted-context
predictions/indices, elapsed CPU time and artifact hashes. No GPU training, paid
compute, new labels, sealed audit access or changes to existing artifacts.

Command: `python -m scripts.train_screen_context`.
Output: C:/CodexArtifacts/pretrained-reuse-r2/groundcua-context-comparison-01.
Runner passes Ruff. This fixed context append experiment is closed negative;
Screen and the overall multimodal quality gates remain unmet.

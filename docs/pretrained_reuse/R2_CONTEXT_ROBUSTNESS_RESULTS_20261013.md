# Context robustness fixed learning comparison

Both predeclared384-update arms completed. Identical initial checkpoint hashes,
same3,072 underlying questions/order,512training states,seed17,batch8,AdamW2e-5,
weight_decay .01,gradient clip1,dropout disabled. Last encoder block/classifier:
1,774,849trainable parameters. Parent initialization is a warm start with optimizer
reset; saved last-block deltas require the pinned base AND parent full-encoder delta.
They are not standalone models or full-resume states.

Validation:128new development states,768questions/context,held-out context templates.
Fixed final checkpoint, no model selection; curves at0/128/256/384 saved. Separate
two-update smoke preceded both runs. Frozen tensor hashes and exact final reload
pass. Full final candidate reversal logits match within1e-5 absolute/relative.

| Context | Plain-training Acc/NLL | Context-training Acc/NLL |
|---|---:|---:|
| Plain | .923177 / .192809 | .915365 / .206184 |
| Irrelevant prefix | .513021 / 2.043618 | .535156 / 1.260892 |
| Necessary prefix | .899740 / .233104 | .895833 / .241307 |

Irrelevant-prefix Brier/ECE improve .777766/.315995 to .652826/.185967.
Both-correct opposite-condition pairs per384: plain329→324, irrelevant108→115,
necessary313→311. All metrics and curves are in the adjacent JSON; no arms omitted.
The treatment reduces overconfidence on irrelevant prefixes, with modest Accuracy
gain and small plain/necessary trade-offs. It remains far below the quality goal.
No product adoption, model-family ceiling claim, or automatic budget extension.
This is a single-seed synthetic development comparison, not general language or
multimodal quality. Paired state-level uncertainty remains to be computed before
claiming a resolved gain. Rolling training CE is not a fixed-example generalization gap.

GPU local RTX3080Laptop, FP32, frozen parameters verified. Timed training/evaluation
sections65.518/76.446seconds respectively, excluding model startup; CUDA allocator
peak is recorded per result (second arm inherits process-wide maximum). No cloud
cost. Outputs `context-robustness-{plain,context}-run-01` and separate smoke outputs
under C:/CodexArtifacts/pretrained-reuse-r2. Initial/final weights, input IDs, config,
gradient norms, CE history and every scheduled prediction are retained.

Command: `python -m scripts.train_context_robustness` (source1f55c31).
Corpus/sampler tests4passed; Ruff passed. Corpus and parent checkpoints untouched.
Full CPU suite/CI for the final report commit must still be checked; neither is
implied by the successful GPU experiment.

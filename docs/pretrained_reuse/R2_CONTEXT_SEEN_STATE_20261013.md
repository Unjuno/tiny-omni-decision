# Fixed seen-state context diagnostic

Inference only with both final384-update checkpoints. First128 training states in
corpus order, all three views (2,304rows). Some views were not presented to each
arm, so predictions explicitly record consumed IDs. This is training-side diagnosis,
not a new independent benchmark or checkpoint selection.

| Training-state view | Plain-trained Accuracy | Context-trained Accuracy |
|---|---:|---:|
| Plain | .933594 | .928385 |
| Irrelevant context | .511719 | .536458 |
| Necessary context | .917969 | .910156 |

For context-trained weights,238 actually consumed irrelevant-context rows reach
.550420 Accuracy,1.077605NLL. The530 unconsumed views of seen states reach.530189,
1.114762NLL. All768 irrelevant views have1.103248NLL. Corresponding development
Accuracy.535156 and NLL1.260892 use different context templates and unseen states;
do not call their difference an identically distributed generalization gap.

The weakness also occurs on consumed training examples. Therefore unfamiliar
development states/templates alone do not explain the result. This supports an
unresolved fitting/optimization/representation limitation under this short final-
block intervention, not a unique capacity diagnosis. More layers or more budget
would require a new controlled comparison; neither is automatically justified.
No epochs, prompts or examples were selected to improve these results.

The corpus and parent/final checkpoint hashes were checked before inference.
All logits finite, no truncation. No training or GPU-process intervention.
Outputs: C:/CodexArtifacts/pretrained-reuse-r2/context-seen-state-diagnostic-01.
Runner: C:/CodexArtifacts/pretrained-reuse-r2/evaluate_context_seen_states.py;
its hash is saved in protocol.json. Full probabilities, consumed flags, per-context
Accuracy/NLL/Brier/ECE and source hashes are retained. This does not qualify an
Omni product or alter previous evaluation results.

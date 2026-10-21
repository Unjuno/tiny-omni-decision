# FLAN fitting diagnosis and decoder-scope comparison

First replayed the last-block final checkpoint on all3072 consumed training
records: Accuracy.257161,NLL1.399504,Brier.756705,ECE.048109. Development was
.239583/1.407907. The small gap and poor training fit do not support severe
overfitting as the main explanation. They do not identify a unique cause.
Replay used no updates and reproduced the first16 saved validation predictions.

Then completed one controlled broader-scope run from original pretrained weights:
all eight `decoder.block.*` layers train; encoder, input/output embeddings and
final norms remain frozen. Same seed17, example-order hash,3072unique examples,
512states,384updates,batch8,AdamW2e-5/.01,clip1,dropout off and validation rule.
Initial validation probabilities are bitwise equal to the last-block reference.

| Step | All decoder Accuracy | NLL |
| --- | ---: | ---: |
| 0 | .261285 | 1.576577 |
| 128 | .250000 | 1.408599 |
| 256 | .235243 | 1.404831 |
| 384 | .243056 | 1.392757 |

Trainable count increases3,147,264→25,178,304 while resident count stays76,961,152.
No meaningful task-quality gain is established; close decoder-scope expansion
in this budget/format. This does not demonstrate a universal model capacity
limit or justify extending steps, unfreezing encoder, or product adoption.

Final run time45.33s excluding load, GPU peak730,959,872bytes, no cloud cost.
All frozen tensors remain equal to original. Earliest minimum validation NLL
selects384; best/final files retained separately, best reload exact for first16.
No sealed set or transfer benchmark accessed. Previous artifacts untouched.

Commands: `python -m scripts.replay_flan_context_train` and
`python -m scripts.train_flan_decoder_context`.
Artifacts under `C:/CodexArtifacts/pretrained-reuse-r2/`:
`flan-last-block-train-replay-01/` and `flan-decoder-context-01/`.
Companion JSONs preserve metrics and full-run artifact hashes; source scripts
retain fixed prompt, option masks, protocol and example-order hashing.

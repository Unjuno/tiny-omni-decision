# Controlled FLAN serialization comparison

Completed a fresh pretrained initialization run with only input serialization
changed: remove `{` and `}` from the state string at prompt construction. No
facts, field names, candidate IDs, options, targets or corpus files changed.
All train/evaluation batches assert zero UNK tokens and no truncation.

Same decoder-all-blocks training, seed17, order SHA,3072unique examples/512states,
384updates,batch8,AdamW2e-5/.01,clip1,dropout off,validation schedule and selector
as the preceding run. Protocol comparison differs only in serialization and
source hash. This is an observed development diagnostic, not a blind test.

| Step | Brace-free Accuracy | NLL |
| --- | ---: | ---: |
| 0 | .254340 | 1.562701 |
| 128 | .243056 | 1.410152 |
| 256 | .230469 | 1.405099 |
| 384 | .243924 | 1.394287 |

Original-brace final results were Accuracy.243056,NLL1.392757. The new run gains
only two net correct answers out of2304 while NLL slightly worsens. This does
not support brace UNK tokens as a sufficient explanation for failed learning.
Close this format/scope comparison without further prompt search or budget
extension. No useful task quality or family-wide impossibility is established.

Selected step384 by predefined NLL rule. Frozen tensors are unchanged; best
reload matches first16 predictions exactly. Best/final, all step training losses,
four validation predictions and per-context metrics are retained separately.
Time45.67s excluding load,CUDA peak730,186,752bytes,local only. No prior artifacts
or data modified. No sealed-set or transfer evaluation.

Command: `python -m scripts.train_flan_no_braces`.
Artifacts: `C:/CodexArtifacts/pretrained-reuse-r2/flan-decoder-no-braces-01/`.
Companion JSON includes run metrics, paired correctness changes and artifact
hashes. Ruff passes; preceding audit commit c395fd3 has successful GitHub CI.

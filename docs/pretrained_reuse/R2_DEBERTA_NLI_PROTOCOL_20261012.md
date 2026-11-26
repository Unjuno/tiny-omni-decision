# Compact NLI fixed condition comparison

Use the exact existing256-question/64-state diagnostic and hypothesis renderer
from evaluate_nli_condition_probe. No rewording, candidates, numbers or labels
changed. Input hash8737946632cf65dd85b6360e9ed1f7b7b70ed7b59df230a0a7f5c9698233fb74.
Onlycheckpoint changes to pinned DeBERTa xsmall70,830,722 parameters audited inR1.
Use its verified entailment label0; previousMiniLM2NLI uses label1. Same scoring:
softmax of the three candidate entailment logits,temperature1. Do not try margins
or probability calibration if results are poor. SameFP32 and batch32pairs.
No training, checkpoint selection, prompt search or new audit. All256 retained.
Save Accuracy/NLL/Brier/ECE, all four conditions, opposite-condition both-correct,
full probability rows, weights/input hashes, parameter count,VRAM and seconds.
Separate output:deberta-xsmall-condition-probe-01. The old NLI reference remains.
This is a constructed, already observed diagnostic, not a blind quality result.

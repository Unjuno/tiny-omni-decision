# FLAN training input audit

Read only the hash-frozen9216 training records; no model scoring, validation or
data changes. Independently recomputed every extremum target from state numeric
fields and the recorded oracle direction:9216/9216 agree and are supplied options.

Reproduced the training prompt/tokenizer path. Every prompt contains eight UNK
tokens, corresponding to the four JSON objects' opening/closing braces. Removing
only braces **for tokenizer inspection** yields zero UNK in every record. The
actual corpus/prompts and model artifacts were not changed or reevaluated.

- Maximum175tokens; zero truncation at512.
- 9216distinct prompts remain9216distinct token sequences; no whole-prompt
  collision found.
- A/B/C/D labels are distinct single tokens71/272/205/309.
- Field names, numeric values, condition and candidate IDs remain represented;
  the sample decoded prompt confirms braces alone appear as UNK.

This identifies a serialization weakness, not proof that it caused failed
learning. Correct labels and nontruncated distinct inputs do not establish that
the pretrained representation encodes useful numeric comparisons. Any future
serialization experiment must preserve all facts/options and be a separately
frozen controlled run; do not reinterpret earlier results as successful.

Command: `python -m scripts.audit_flan_train_tokens`.
Artifact: `C:/CodexArtifacts/pretrained-reuse-r2/flan-tokenizer-train-audit-01/result.json`.
Full existing CPU suite:730passed,1skipped (14.22s). That suite is a software
regression check, not evidence of model quality or coverage of new GPU experiments.

# TRAIN-side semantic teacher prerequisite

Frozen before any teacher forward, 2026-10-15. This implements strategy section11;
it is an inference-only screen, not a student training experiment or release audit.

## Question and model

Can the existing pinned instruction model solve condition meaning from correct
structured observations, before asking the small Decision model to imitate it?
Use only pretrained Gemma4 E2B IT QAT-unquantized weights, not a legacy adapter.
Revision `6befbaca7398925921802abd1f277b495b78b738`, expected weight SHA256
`33fe0cece08fb527ffefbd1a3a9ce73bd71073727993a283506293e5c6bf0137`.
No quantization is performed: this is the upstream half-precision QAT export.
All resident parameters are counted; approximately5.1B is a teacher reference,
never a <=200M product candidate.

The pinned [official model card](https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-unquantized/blob/6befbaca7398925921802abd1f277b495b78b738/README.md)
labels these weights Apache2.0 and links to Google's
[license text](https://ai.google.dev/gemma/apache_2). Existing locally cached
weights are reused read-only; no paid services or dependency upgrades.

## Cohort and controls

Read only the two hash-pinned TRAIN files in the config. No validation or sealed
audit data is loaded. Hash(seed17, source, group) selects32 complete groups from
each source before predictions. This is known TRAIN material, not unseen quality.

- Binding:32 states ×4 properties ×5 conditions =640 prompts. Keep all40 typed
  choices and targets. Use one descriptive candidate wording throughout. Conditions:
  explicit target, exclusion of the other three, cancelled request/replacement,
  reverse exclusion choice order, and empty-question control.
- Numeric:32 states ×8 questions ×2 candidate orders =512 prompts. Keep the
  original questions, four/five choices and labels: min/max cost/risk/time,
  risk-threshold count, and minimum cost among threshold-eligible items.
- Total1,152 prompts /64 independent source-qualified states. Numeric time order
  is ordering numbers, not a native-video capability test. Empty-question repeated
  rows are deliberate controls, not additional independent observations.
- Labels and prompt are programmatically verified. Oracle metadata/IDs/targets
  never enter the prompt. Source state and question are the actual inputs.

## Fixed inference and resource budget

Use shipped chat template, thinking disabled, batch1, BF16, SDPA math, deterministic
seed17, no dropout/gradients/optimizer. Forty case-sensitive single-token labels
`A..Z,a..n` are validated as continuations of every complete prompt. Conditional
softmax temperature1; also save total full-vocabulary mass on the allowed labels
and the unrestricted top token. No prompt/temperature/seed selection from results.
No truncation; fail if any processed prompt exceeds2,048 tokens.

Windows safetensors `pread` is selected only inside this new process's loader;
the installed libraries and historical code are unchanged. Missing/mismatched
model weights fail closed. Record unexpected checkpoint keys and loaded count.
Use only idle local RTX3080Laptop16GiB. Guards:40min per process,15GiB allocated
VRAM,24GiB process peak host memory,64MiB output and8GiB disk reserve.

Prepare full rows/tokens/hash pins first. Separate smoke uses the first hash-picked
group from each source:36 prompts covering every arm. Continue to the full fixed
cohort only if smoke mechanics and projected runtime fit. Smoke predictions do
not change the protocol. Fresh-process replay checks the same predeclared36 rows
against the full run: identical IDs/predictions and max absolute logits<=1e-5.
This is bounded replay, not verification of every full-run prediction.

## Reporting and next decision

Save every label logit/probability/prediction/target, raw allowed-label mass,
Accuracy/NLL/Brier/ECE by condition/task, macro metrics, distinct-target condition
pair joint correctness, candidate-order agreement and probability differences.
Report load/inference/wall times, memory, exact versions, all code/input hashes.
The existing small-model validation scores are context, not same-cohort teacher
comparisons; do not calculate improvements by subtracting those scores.

TRAIN screen eligibility is task-specific:Accuracy>=95% and ECE<=.10 in each
relevant explicit/exclusion/switched/numeric task; reversal answer agreement>=99%
is reported separately. This is a conservative prerequisite for considering a
future independent student experiment, never generalization or product approval.
Show failures without removing difficult groups. All prediction sets remain.
If conditional semantics fails, do not begin distillation or claim a capacity
ceiling. If it succeeds, prospectively freeze independent syntax/state transfer
and compare labels-only versus option-logit supervision with equal student budget.
Native-input and multimodal quality still require separate evidence either way.

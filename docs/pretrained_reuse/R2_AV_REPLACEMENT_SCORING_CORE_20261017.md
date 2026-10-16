# AV replacement scoring core

Implements the inference loop in the already frozen
`R2_GREATEST_HITS_INPUT_REPLACEMENT_PROTOCOL_20261017.md`. This is a separate
development branch from the live `codex/r2-av-transport` cache/audit checkout.
No running source, training condition, dataset, checkpoint or artifact is changed.

`scripts/greatest_hits_replacement_scoring.py` takes a model, admitted event store,
validation metadata, frozen inventory and output callback. It rejects an inventory
that differs from label-blind reconstruction and checks store option identity.
It emits natural, self-audio, self-video, donor-audio and donor-video results for
every frozen pair in inventory order. Every record contains original ID, planned
donor ID, actual input donor ID (null for natural), task/date/parent, options, full
unpadded logits and candidate gate values where available.

Self replacements must reproduce both natural logits and gates exactly. Optional
natural reference arrays are keyed by question ID and must match exactly, including
option count. The production caller must supply selected-checkpoint reference
arrays after validating their metadata/option order. Finite logits and active,
finite, bounded gate arrays are mandatory. Resource guards run before each forward.

For each donor view, report per-question flips, total variation, Jensen-Shannon
divergence (nats), confidence change, and question-weighted overall/task/date/parent
aggregates. These describe sensitivity, not counterfactual correctness. The function
does not read target labels, optimize, calibrate, select checkpoints or compute
Accuracy/NLL/Brier/ECE on mismatched inputs. Correlated questions are not independent
trials; these summaries are descriptive, without inferential confidence intervals.

## Verification scope

Tests use all five actual readout architectures on deterministic synthetic native
token blocks with two option counts. They check all five views, exact self swaps,
label/order invariance, unchanged state and absent gradients, inventory rejection,
natural-reference identity/logit checks, option-binding rejection, nonfinite/short
logit rejection and propagation of resource-guard failures. Initial tests failed
because the module was absent; the additional binding/actual-donor checks also
failed before their implementation. Existing transport tests remain intact.

Local verification: Ruff passed; full CPU suite 2,403 passed, 1 skipped in
73.39 seconds; all 25 existing manifest validation commands passed. Logs and exact
commands are preserved under
`C:/CodexArtifacts/pretrained-reuse-r2/greatest-hits-replacement-scoring-verification-01/`.
No GPU execution or development-validation model scoring was performed here.

## Remaining execution gate

This is a scoring **core**, not a guarded production CLI or a trained-model result.
The production entry point still needs selected-run/checkpoint/input/inventory hash
admission, exact natural-reference ID/option alignment, fresh immutable output paths,
before/after source/input/checkpoint verification and terminal failure receipts.
Do not invoke it on development validation until matched model selection is frozen.
Do not use initial or smoke models as selected-model quality evidence. Full frozen
validation scoring for all five selected arms remains pending.

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

## Selected-model entry point

`scripts/score_greatest_hits_replacements.py` adds the guarded execution entry point.
It requires `--selection-set`, `--selection-set-sha256`, `--audit`, `--arm` and
`--max-seconds`. The selection JSON maps exactly C0/E0G0/E0G1/E1G0/E1G1 to objects
with `path` (completed matched run directory) and `result_sha256`. Construct and
hash this immutable selection record only after all five runs finish selection.
The CLI refuses smoke/incomplete runs, missing arms, differing training sources,
runtime versions, inputs or non-arm training configuration. It does not select
a model itself or accept an alternate checkpoint override.

The selected run's complete artifact ledger, best checkpoint/step/config/sampler,
input audit and frozen inventory are verified. Shared executed Python files and
the training protocol must match the selected run's source hashes in the diagnostic
checkout. Natural reference arrays are aligned by ID with exact option order and
padding checks, then compared exactly during inference. The original common TRAIN
and validation payloads are not re-evaluated by this diagnostic.

Each arm writes once to
`C:/CodexArtifacts/pretrained-reuse-r2/greatest-hits-selected-replacements-<ARM>-01`.
An existing destination is rejected, including a failed attempt; no automatic
retry, deletion or overwrite is provided. Incomplete attempts retain error and
traceback receipts. Predictions are streamed as JSONL, all 44,375 forwards and
17,750 exact self checks are required, and source/input hashes are verified again
before success. Resource allowance is explicit, with 10 GiB host/6 GiB GPU guards
and 16 GiB free-disk reserve. Occupied GPUs are rejected before model allocation.

CPU tests cover selected-run/config/sampler binding, completed-five-arm admission,
natural-reference identities/options/padding, and actual CLI rejection receipts
and output preservation. CLI rejection tests stub only Git metadata; real fixture
receipt hashes are checked. The previously described scoring tests use actual
readout architectures. The full admitted CLI path with real selected checkpoints
and CUDA has **not yet run**; no end-to-end diagnostic result is claimed.

Entry-point verification: Ruff passed; full CPU suite 2,412 passed, 1 skipped in
69.20 seconds; all 25 manifest commands passed. Exact commands/logs are under
`C:/CodexArtifacts/pretrained-reuse-r2/greatest-hits-replacement-cli-verification-01/`.

Do not invoke it on development validation until matched model selection is frozen.
Do not use initial or smoke models as selected-model quality evidence. Full frozen
validation scoring for all five selected arms remains pending.

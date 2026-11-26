# Fixed conditional representation / factorization screen

Prospective next step after 8dcb4ab. No training, calibration, template search,
checkpoint selection, or sealed-audit access. Prior conditional-contrast and
FLAN/SmolLM/NLI negative results remain unchanged. This screen asks whether a
pretrained NLI representation and explicit separation of request relevance from
state truth make the conditional field error smaller before any new fit.

## Fixed design

Two existing checkpoints: retained 22,713,601-parameter MiniLM KL512 (base revision
233902d25c440f23af6f7d6e94d2946bac0bee0a, delta SHA c87ddf2197e276814e4941efb7ad7431d416316eb786ea59b8331925ca16ab02)
and unmodified 70,830,722-parameter DeBERTa xsmall zeroshot (revision
262ae02f29173eec1c250f90804dc7edc677dcff, MIT per existing pinned audit).
The latter uses its native entailment logit index 0; MiniLM uses its scalar logit.
Both use FP32/eval/frozen parameters, temperature1, fixed batch32, no truncation.
No claim that scores are already calibrated; measure NLL/Brier/ECE15.

32 previously unused combinations of four existing CLEVR4 attribute vocabularies,
hash-ordered with seed17. Exclude every state in the earlier TRAIN labels,
semantic bridge, binding corpus, independent-transfer corpus, language corpus and
latest contrast corpus. Store exact source hashes and excluded identities.
Eight fixed syntax families x32states x4requested attributes =1024questions.
All40 candidates (10values per field) remain; no target-conditioned candidate pruning.
These are synthetic structured inputs, zero native media. Novel states do not imply
unknown real-world concepts or blind product evaluation. All eight families are
declared together, executed once, reported individually with no favorable subset selection.

Readout comparison within each checkpoint:

- Direct: premise `Record: <JSON>\nRequest: <condition>`; hypothesis
  `The correct response to the request is: <candidate text>`; softmax all40 logits.
- Factorized: request premise paired with `The requested property is <field>.`
  yields four field logits. Record premise paired with each candidate fact yields
  40 value logits. Combine log softmax(field) with log softmax(values within field).
  Sum of all40 output probabilities is1; schema fields are public input.
  Gold task/rule/target metadata cannot enter rendering or scoring. No programmatic
  parsing of the request to recover the correct attribute is allowed.

Both checkpoints use the same strings and candidates. This is an interface screen:
the direct rendering differs from the historical MiniLM query/state pair, so its
score is not a reproduction of that earlier benchmark. Across models, pretraining,
size and historical adaptation differ; do not attribute a gain only to capacity.
Deduplicate identical rendered pairs exactly, preserve indices and account for
requested/unique forwards. The values and request factors can be reused explicitly.

Controls: re-score all reversed candidates, align by public IDs, report maximum
probability delta/agreement; swap each record for the next record and score against
both original and donor truth; remove the request and retain the deliberately
ambiguous four-question targets. No-question results do not impose a mandatory
text front end on native tasks. Save per-question IDs/targets/predictions/probabilities,
all raw factors, source/code/config pins, actual parameter counts and resource use.
Run an independent fresh process replay of all outputs; weights must remain exact.

## Decision and limits

Execution requires full counts, all identities checked, finite scores, all40
probabilities, unchanged weights and full fresh replay (absolute tolerance1e-6).
Discrete prediction IDs/counts must match exactly, including order-control ties.
Research continuation requires at least3pp better joint40-way Accuracy versus
the same-model direct arm, a positive state-paired95% interval (10000 resamples,
seed17, percentile), minimum-family Accuracy >=90%, and no family >1pp worse.
NLL/Brier must not increase; ECE increase <=.02. Evaluate both candidates in full.
The interval resamples states, not syntax families; repeated fixed syntax does
not support confidence claims over arbitrary unseen natural-language requests.
These gates are not product95–98% or model-selection permission.
If neither passes, close this fixed rendering/decomposition screen; do not sweep
prompts, temperatures or coefficients on it. Numeric reasoning and original task
retention are still required before any adoption/new semantic fit. Their earlier
failures remain evidence; this attribute screen cannot supersede them. Do not
repeat the old256question numeric probe or expose final audit to fill this gap.

Whole-budget scenarios use measured component counts: VideoPrism114,365,184,
Whisper encoder8,208,384, environment audio1,961,239. Adding DeBERTa gives195,365,529
before all fusion/temporal/readout weights (4,634,471 remaining); with full Whisper
it gives224,917,785 and fails. Encoder-only speech quality is unproven, so neither
scenario is an admitted Omni product. Existing package/temporal/joint gates stay open.
The MiniLM scenario is147,248,408 before heads, with the same speech caveat.

Run local GPU only while free. Maximum each evaluation900s, CUDA allocation8GiB,
host working set6GiB, new output512MiB, disk reserve2GiB. Separate new prepare,
smoke, primary and full replay directories. On a resource/numeric/provenance failure,
retain the failed output and report incomplete; never alter the frozen experiment.

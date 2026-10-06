# Postquant teacher-cache canonicalizer

## Status

This stage implements the identity and serialization boundary for the frozen Q0
Recovery Teacher option-logit cache. It does not run Gemma, merge LoRA, quantize
a model, start Recovery/Re-specialization, or touch a sealed/final audit.

The input to scripts/build_postquant_teacher_cache.py must already contain
enriched rows with exact ordered options, labels/token IDs, Q0 option logits,
target, and provenance. Legacy prediction files that do not contain the
original ordered option text are not silently upgraded or guessed.

## Required row fields

- sample_id
- ordered_options (2..62 nonempty distinct strings)
- option_labels (same length, distinct strings)
- option_token_ids (same length, distinct integers)
- teacher_option_logits (same length, finite numbers)
- target_index
- temperature (cache v1: exactly 1.0)
- preprocessing_sha256
- model_sha256
- adapter_sha256
- processor_revision
- sampled_frame_policy
- source_media_sha256

validate_teacher_cache_record(record, expected) compares every identity field
and fails closed on changed option text/order, labels/token IDs, target,
preprocessing, model/adapter identity, processor revision, frame policy, or
media hashes. Logits are cache payload and are covered by the cache digest.

## Canonicalization

Run:

    python scripts/build_postquant_teacher_cache.py       --input /absolute/path/q0-enriched-option-logits.jsonl       --output /absolute/path/postquant-q0-teacher-cache

The output path must not already exist. The command writes a sibling temporary
directory and promotes it with os.replace. It never overwrites an existing
cache. Input/output paths containing audit, sealed, or heldout components are
rejected. Duplicate JSON object keys and NaN/Infinity are rejected.

Output:

- teacher-cache.jsonl: canonical, order-preserving JSONL.
- manifest.json: row count, order-sensitive cache SHA-256,
  sample-ID-order SHA-256, and source-JSONL SHA-256.

Duplicate sample IDs are rejected. The cache digest and sample-order digest use
separate domains so the future trainer can compare the exact cache order
against its frozen sampling contract without parsing logits.

## What remains before GPU work

This command does not generate Q0 logits. A later model-aware step must load
the frozen Master in eval/no-grad mode, produce the enriched rows, and verify
Master-plus-Decision-LoRA versus merged-Master equivalence on a fixed
development slice.

The postquant trainer must validate each cache row against the active training
sample before consuming its logits and must implement exact
optimizer/scheduler/RNG/sampler resume. Until that integration exists, the
8-update real-model smoke and Q2/Q3/Q4 training remain blocked.

## References checked

- https://docs.python.org/3.11/library/json.html
- https://docs.python.org/3.11/library/os.html#os.replace

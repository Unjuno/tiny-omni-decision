# Contextual numeric tokenization audit

Read-only CPU audit using the same pinned local tokenizer and exact training
corpus hashes. Each of1,024underlying states counted once; four items x three
numeric fields yields12,288numeric occurrences per corpus. Audit offsets in the
actual `State: <JSON>` segment, not only isolated number strings. Thresholds and
answer options are outside this occurrence count. All tokenizer file hashes and
source runner hash are preserved in adjacent JSON.

| Corpus | Unique values | One-token occurrences | Two-token occurrences | Multi-token fraction |
| --- | ---: | ---: | ---: | ---: |
| Original1..199 | 199 | 12288 | 0 | 0% |
| Augmented numeric range | 583 | 8259 | 4029 | 32.788% |

No unknown-token occurrences. Per-scale two-token fractions:
x:0/3072;2*x+7:297/3072;4*x+31:1661/3072;8*x:2071/3072.
The changes increase vocabulary diversity and introduce composition across
subword pieces. Magnitude, vocabulary and segmentation change together, so this
is not causal proof that fragmentation caused training failure. There is no
input-loss explanation here. Frozen embeddings and single-pass training remain
possible contributors, alongside representation and optimization.

Before adding parameters, the next proposed optimization control is to reuse
the already successful narrow-format encoder checkpoint as initialization for
the unchanged wide-range recipe, with a new optimizer and output. This would
cost additional training and inherit prior training history; it must not be
called equal-total-compute or exact resume. Define the budget and retention
selector before executing. That experiment has not been run in this audit.

No training, tokenization changes, prompt tuning, additional GPU process or
sealed data access occurred. This remains a text development diagnosis, not a
four-modality quality result. Existing checkpoints/corpora remain unchanged.

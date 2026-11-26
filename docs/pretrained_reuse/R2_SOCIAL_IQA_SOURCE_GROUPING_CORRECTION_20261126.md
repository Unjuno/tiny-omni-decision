# Social IQa source grouping clarification — 2026-11-26

## Why this note exists

The existing Social IQa internal readout used a seed-17 split grouped by exact
normalized context. Its report correctly says this does not prove ATOMIC
source-event disjointness. The source paper adds an important distinction:
the publisher's official train/dev/test partitions were constructed so that
contexts derived from an ATOMIC event stay in that event's partition. This does
not make a random split *within publisher train* event-disjoint. The internal
validation used for the MiniLM readout therefore remains potentially exposed
to sibling contexts from the same ATOMIC event as training rows.

## Pinned primary-source evidence

The Social IQa paper states that contexts were assigned to the same partition
as their originating ATOMIC event (Sap et al., 2019, §3.5). It also describes
the annotation pipeline: three workers validate training tuples, while dev and
test receive a second five-worker validation and lightweight adversarial
filtering; human accuracy on separate random subsets was 87% on dev and 84% on
test. These are task-specific properties, not a general multimodal quality
target.

Primary source: [Social IQa: Commonsense Reasoning about Social
Interactions](https://aclanthology.org/D19-1454/), especially §3.4–3.5.

## Consequences for existing evidence

- Keep the existing internal validation numbers as exploratory model-selection
  evidence, but do not describe that random context-group split as
  ATOMIC-event-disjoint.
- The one-shot official-dev audit remains a distinct publisher-held-out result;
  it was not used to fit or select the checkpoint. It is already exposed now
  and must not be tuned against or replayed.
- Do not retrofit the internal validation labels or select another checkpoint
  based on official dev.
- A future Social IQa comparison would need an untouched admitted task or a
  newly frozen source-event-grouped split from eligible training-only records.
  The author page advertises a supplemental release with ATOMIC dimensions and
  answer-source fields, but its per-row source-event identifier and license
  must be verified before relying on it. No supplemental archive was fetched
  or inspected for this clarification.

This correction narrows the source-overlap uncertainty; it does not improve the
existing model result or close any R2 quality gate.

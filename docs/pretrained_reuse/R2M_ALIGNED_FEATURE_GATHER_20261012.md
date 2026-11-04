# Time-indexed feature gathering into fusion

`gather_aligned_features` connects the existing `ObservationBatch` causal
hold-last alignment to numerical features usable by `FusionDecisionScorer`.
It is an implementation connection, not a new model-quality experiment.

## Cache contract

Each configured source points to a feature NPZ with exactly `values[T,D]`,
`timestamps_ms[T]` and Boolean `valid_mask[T]`. The observation SHA256 covers the
whole NPZ. Read/hash/decode operate on the same bytes. Feature dimensions, times
and masks must match observation metadata; valid features must be finite.
Invalid rows may contain nonfinite placeholders and never reach the model.
Indexed raw payload references are rejected by this particular cache adapter.

The caller supplies one distinct source ID per modality and the expected width.
For each query it receives fixed-width features, explicit presence, selected
row indices and ages. Unknown/absent configured sources, future-only evidence
and stale evidence produce zero storage plus a **false** mask. The zero vector
alone never establishes presence. Extra observations are not silently averaged
into a selected source. Shared-clock verification remains mandatory.

Before inference, the caller must apply the task's required-evidence gate to
these masks. The model's ability to handle a nonempty subset is not permission
to answer a task that requires a missing source. The integration test verifies
this gate skips the model entirely before evidence arrives and after expiry.

## Temporal meaning and limits

Times must denote when all evidence used by an encoder became available, plus
any relevant processing delay for an online deployment. The helper validates
consistency, not the truth of declared clocks or receptive fields. An encoder
that attends over an entire video may produce frame-indexed tokens that all
depend on later frames. Those tokens cannot be treated as causal at individual
frame timestamps merely because token indices look temporal.

The existing VIDIMU whole-episode features are unchanged. Their unresolved
synchronization is not fixed by this adapter. No observed video/IMU offset was
invented, no existing feature timestamps rewritten, and no sealed data opened.
This implements causal feature selection/hold-last, not temporal reasoning,
interpolation, synchronization estimation or a learned latent-state update.

## Verification

Five tests execute real NPZ serialization, SHA verification and selection:

1. Before/at/after evidence, invalid rows, inclusive freshness and absent sources.
2. Modified payload hash and inconsistent timestamps rejected.
3. Wrong modality and duplicate source binding rejected.
4. Perturbing a future feature leaves earlier gathered features unchanged.
5. Gather -> actual Torch fusion scorer -> evidence gate -> deterministic JSON;
   only the fresh-evidence row calls inference, with no mandatory text criterion.

These are fixtures, not held-out quality measurements. The next real streaming
experiment must establish encoder availability times and task labels before
reusing this path for accuracy or latency claims.

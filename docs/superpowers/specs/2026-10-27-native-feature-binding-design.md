# Native feature handoff for text-free Decision JSON

The R2 roadmap leaves audio/image/video feature-provenance handoff open. Existing
tensor-level `decide_native` methods validate request semantics, but do not bind
the input tensor to the request's actual media or declared preprocessing.

Use the existing immutable `ObservationFeatureCache` and full
`ObservationFeatureKey` through a new explicit `decide_bound_native` entry point.
An application-controlled extraction receipt fixes the full key, feature-payload
hash, observation identity, clock and sample timestamps. Request metadata cannot
replace this receipt. Every complete decision verifies current media bytes,
loads the exact cache identity, checks the separately pinned feature hash and
float32 shape/finiteness, then calls the existing readout without fitting.

This is an implementation choice within the user's authorization to proceed.
Passing an unchecked caller tensor is retained as a low-level API, not used as
evidence of binding. Running an encoder afresh on every decision would discard
the existing observation-cache benefit; fresh-encoder parity remains a separate
test. A receipt is trusted producer evidence, not cryptographic proof that an
untrusted producer executed an encoder.

Keep original probability/option/tie semantics. Add `feature_provenance` only to
the verified entry point's selected result; incomplete inputs abstain without
claiming a cache was loaded. Indexed media containers fail until they have an
explicit per-index contract. Screen, sensor and fusion use separate paths.

Acceptance evidence: reject stale media even if stat metadata matches, reject
cross-clock/time/source/key mismatches and forged cache bytes, preserve native
readout results and option reorder semantics, and exercise existing real audio,
image and video readouts/cache rows on CPU with every legacy artifact read-only.
No new quality, calibration, encoder execution, GPU, or sealed-audit experiment.

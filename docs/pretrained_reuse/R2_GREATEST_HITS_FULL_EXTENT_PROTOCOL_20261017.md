# Fixed full-extent VideoPrism execution before common AV caching

Use exactly the 21 TRAIN events in the source-review plan SHA256
`951d1078d3fae228256bb398bd636fea2393fd7aa4b05969ef04655c7f475ae5`.
Read the saved original RGB/PTS only. No new sample selection, labels, annotation
inspection, validation/audit loading, learning or quality scoring.

The existing event sampler selects eight distinct actual frames spread across
the same half-second interval. Reuse the existing full-extent bilinear policy:
456x256 -> 288x162, centered with 63 black rows above/below on a 288-square canvas.
No central crop, gain, spatial augmentation, retiming or source-specific geometry.
Positions retain actual sampled PTS on the shared event clock; spatial coordinates
refer to the encoder canvas including padding. Padding tokens remain in the
contextual encoder output, so they must not be described as absent observations.

Load the same pinned FP32 VideoPrism base in the existing CPU JAX/TF environment;
verify the 114,365,184 parameter inventory and frozen state before/after all forwards.
Save each prepared input, 2048x768 feature array, 2048x3 positions and selected PTS;
check finite values and exact serialization. Save a contact sheet from the prepared
eight frames for inspection of known edge-contact cases. Run a separate process on
the same pins/versions and require exact replay of all outputs. Record load/forward/
total time, peak host memory and artifact hashes. Keep all old results unchanged.

Use fresh output directories. Cooperative guards: 900 seconds, 8 GiB process peak
working set and at least 2 GiB free disk; check between forwards and at completion.
No GPU, new environment or cloud compute. Unexpected exceptions remain recorded
as incomplete; do not silently change geometry, frames or dtype to force success.

Passing means this concrete input/encoder path is reusable for the scoped native
development comparison, not that it improves quality or establishes physical AV
synchronization. After passing, prepare the complete immutable common cache and
measure its actual throughput. The Attention and exchange/gate arms must all use
that same cache. The original Attention architecture and earlier results remain
the baseline; old cropped scores are not a causal control for the new gate.

The source review admitted material/action/reaction development only. Both-required
semantic truth, valid contradictory/swap labels, actor/site independence, retention,
four-modality 95–98% and product adoption gates remain open.

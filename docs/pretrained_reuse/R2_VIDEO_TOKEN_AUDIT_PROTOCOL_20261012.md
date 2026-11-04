# Unpooled VideoPrism token feasibility

Read-only frozen backbone audit, no training or quality evaluation. Select the
first two unique training media in the existing256-scene manifest order. Pin
VideoPrism source/checkpoint, manifest and old pooled cache hashes. Keep existing
8 endpoint-inclusive frames,288 resize/crop,FP32 and working JAX/TF environment.

Upstream FactorizedEncoder explicitly maps `(bn)td -> b(tn)d`; patch_size18 at
288 gives16×16 spatial positions. Validate actual output[1,2048,768] and preserve
as[8,16,16,768] without averaging or dtype change. Unit-test index mapping.
Save FP32 arrays in a new directory, reload byte-exact, check finite values,
compare direct global mean to old pooled feature at atol1e-5, and repeat the
first forward to measure deterministic equality. Do not relax failed gates.

Record parameter count, source/model/media hashes, environment/backend, actual
decode/preprocess and forward times (first includes compilation), process RAM,
array/file bytes and projected456-scene footprint. Projection is not an executed
full extraction. Audit input scenes only; no validation/audit quality or temporal
reasoning claim. Do not start full extraction automatically if storage/runtime
is impractical; use measurements to design the next bounded readout experiment.

# Unpooled VideoPrism token feasibility result

PASS_FEASIBILITY_ONLY on two preselected training scenes19/37. No training,
validation scoring, final audit, or claim of better temporal reasoning.
Source `dc82b70cf592502f888a8136832743b9326705e4`; exact command/model/source/media hashes
and versions in companion JSON. Existing upstream tracked source unchanged;
pre-existing build/egg-info/pycache directories were left in place.

## Measured representation

Pinned VideoPrism-B has114,365,184 parameters. At the unchanged8frames and
288×288 preprocessing it returns FP32[1,2048,768], exactly reshaped as
[8,16,16,768]. Upstream `(bn)td -> b(tn)d` establishes time-major indexing;
the index mapping is unit-tested. No pooling or precision conversion in saved arrays.

Both arrays reload exactly. Their global means equal the existing pooled
reference with maximum difference0.0. Repeating the first forward returns
exactly identical tokens. The saved arrays also match the first attempt's hashes.
These are contextualized grid tokens, not verified object tracks.

## Runtime and storage

- Backendcpu, device['TFRT_CPU_0']; no GPU execution claim.
- Model load0.307s; first forward including compilation 6.059s.
- Second scene forward1.056s; repeat forward1.046s.
- Model/audit timer9.292s excludes dependency imports.
  Config-start to finish11.842s includes those imports but excludes
  interpreter startup and earlier provenance checks; not a full CLI benchmark.
- Windows peak working set4,155,723,776bytes; final RSS1,906,651,136bytes.
- Each raw array6,291,456bytes; NPY file6,291,584bytes.
- Projected456-scene NPY footprint2,868,962,304bytes
  exceeds observed free disk2,184,622,080bytes. This projection
  is not an executed full extraction. No full cache was started.
- Local compute only,cloud cost$0. Only two training assets were read.

## Failed attempt and correction

Attempt01 extracted both arrays and passed the earlier checks, but failed
while writing final metadata because this Windows installation does not
provide a distribution named tensorflow. Its total/repeat timings were not
persisted and remain UNKNOWN. Partial outputs were retained unchanged.
Use the imported tf.__version__ instead of assuming a distribution alias;
regression test covers this case. No packages were installed or upgraded.
Attempt02 ran afresh in a new directory and completed all gates.

## Next bounded step

Token extraction is practical in this tiny audit; quality is unmeasured.
A full FP32 file cache does not fit current free disk. Prefer a measured
bounded transfer to a RAM cache shared by all questions for each scene,
rather than duplicating tokens per question or silently lowering precision.
Measure that memory/transport path before a full token-readout experiment.
Keep frame count and encoders fixed; no basis yet for full fine-tuning.

Artifacts remain at `C:/CodexArtifacts/pretrained-reuse-r2/videoprism-token-audit-01`
and `videoprism-token-audit-02`. Large arrays stay outside Git. Four-modality
95–98% remains unmet; this audit supplies an execution path, not quality evidence.

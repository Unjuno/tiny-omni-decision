# Fusion source readiness and causal alignment

This stage implements metadata alignment, not a trained fusion model. The helper
uses latest valid evidence at/before each query, within an explicit inclusive
maximum age. No interpolation, payload reads or inferred clock correction.
Invalid observations may fall back to an earlier valid sample only within that
age limit. Missing/stale entries have None indices and false availability, not
zero-valued evidence. Different clocks fail closed; callers must align clocks
upstream. For causal decisions, timestamp each feature when all its evidence is
available (audio-window end, for example). Equal clock strings alone cannot prove
physical sensor synchronization. Validate source acquisition metadata separately.

## Real paired source candidates: no media ingested

- [UP-Fall official site](https://sites.google.com/up.edu.mx/har-up/) documents
  synchronized camera/sensor trials and timestamps for17 subjects. Public access
  is stated, but a clear dataset license grant was not located on the inspected
  page. The paper's CC-BY license is not assumed to license its dataset. REVIEW.
- [UTD-MHAD official site](https://personal.utdallas.edu/~kehtar/UTD-MHAD.html)
  describes861 trials from8 subjects with RGB/depth/skeleton/inertial channels,
  including recorded timestamps. Need explicit reuse terms and actual downloadable
  timestamp fields before admission; do not assume AVI frame order establishes
  alignment. REVIEW. Unrelated similarly named UTD text-description data cannot
  supply this dataset's license.
- [UR Fall official site](https://fenix.ur.edu.pl/~mkepski/ds/uf.html) specifies
  CC-BY-NC-SA4.0 and approximate nearest-timestamp camera synchronization. This is
  not an unrestricted product-training source. Do not ingest into the current
  approved corpus.

None closes the paired data gate. No whole-goal blocker is declared: alignment
tests and source validation can progress while actual real fusion remains unmet.
Existing STARSS23 audio/video results remain negative and are not replaced by
these readiness checks. No cloud compute or sealed audit access occurred.

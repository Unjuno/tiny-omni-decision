# Fixed explicit-geometry comparison

Both arms use 134 inputs, identical initial checkpoint hashes, 115,648 trainable
parameters, fixed79 fit /20 observed internal-development videos, seed17,50epochs,
AdamW .001/.01,8pairs/batch,500updates. The control zeros the appended five geometry
coordinates; the candidate uses frozen decoder mask centroids, spread and mass.
All original129 coordinates, question/options, sample order and training budget
are preserved. Neither arm selects an epoch. This is distinct from the earlier
129-input baseline: do not attribute differences against that baseline solely to
geometry, since its random initialization and parameter count differ.

| Metric | Zero geometry | Explicit geometry |
|---|---:|---:|
| Train Accuracy | .822785 | .772152 |
| Internal-dev Accuracy | .450000 | .450000 |
| Internal-dev NLL | 2.963596 | 3.576206 |
| Internal-dev Brier | .867171 | .791962 |
| Internal-dev ECE | .376179 | .373448 |
| Mismatched-video Accuracy | .325000 | .325000 |
| CPU fit seconds | 6.603406 | 4.717583 |

Both exact checkpoint reload checks pass; full curves, all predictions, control
conditions and weights are preserved in separate `perception-slot-{zero_geometry,
geometry}-internal-01` and corresponding `-internal-eval-01` directories under
C:/CodexArtifacts/pretrained-reuse-r2. Adjacent JSON contains artifact hashes.

Paired20-video bootstrap (10,000replicates,seed17,95%percentile,1e-12NLL floor):
geometry-minus-zero Accuracy difference0.0 [-.125,.100]; NLL+.612610
[-.331864,1.761316]. Generic helper direction label normal-minus-mismatch denotes
first-minus-second argument here: geometry is first, zero geometry second.
These are observed-development diagnostics, not a blind evaluation. No sealed data.

Decision: no adoption. Explicit geometry has mixed metrics and no demonstrated
Accuracy gain. Stop this feature-append comparison without coefficient, epoch or
seed refinement. Recoverable position is insufficient to establish useful temporal
condition decisions; identity and condition grounding remain unresolved. This does
not prove that all geometric/temporal architectures fail or license compression.

Run: `python -m scripts.run_slot_geometry_comparison`. Source parent9fdf6c5 plus
the committed runner/input-dimension changes. Helpers retain129 as the default
for existing callers. No encoder weights or old artifacts changed. Cloud cost0.

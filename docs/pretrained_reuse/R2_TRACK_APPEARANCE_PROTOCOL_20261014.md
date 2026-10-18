# Model-track-aligned appearance primitive — before execution

Full Omni objective unchanged. The train-only condition diagnosis shows question
and scene information survive, while unseen-scene quality remains poor. Inspect
the next input-only primitive before another fit: bind per-time appearance to
every existing model-produced trajectory. No added model, parameters or learning.

## Fixed inputs and operation

Reuse all256 original fixed-grid TAPIR tracks at the same32 source frame IDs and
the eight earlier fine-grid training references:10103,10186,10212,10387,1087,
10886,10894,10903. Preserve all original cuts/media/cache/weights. No boxes, masks,
role IDs, question targets or annotations are loaded or used for selection.
No development feature arrays or sealed/audit data are opened.

For each point and time, sample the frozen low32×32×256 and high64×64×128 grids
at the track's estimated normalized x/y, concatenate384channels. Coordinates are
the first128 fields reshaped256×32×4 of the pinned motion bank, not ideal seed
positions or GT. Retain the original coordinates without clipping; sampling uses
TAPIR's half-cell convention and border padding. Separately return a Boolean flag
for coordinates in the closed normalized raster[0,1]×[0,1]. Border features for
outside coordinates are **not** evidence of visible objects or valid correspondence.
Keep every point/time; no support/occlusion-driven deletion of QA examples.

Use FP64 interpolation accumulation then FP32 output, no extra normalization.
Compare to pinned official convert_grid_coordinates/map_coordinates_3d CPU
sampling at identical queries. Predeclared absolute maximum error≤1e-5 (the prior
descriptor gate); repeat/readback/point-permutation gates are strict byte equality.
Clone reference coordinates because upstream map_coordinates_3d modifies its
temporary time coordinates. Verify source arrays/coords are unchanged.

One fixed comparison uses the same grids with each point's first estimated x/y
broadcast over all32times. Actual-track versus fixed-position descriptor changes
are descriptive input measurements, **not** semantic identity or QA improvements.
Record support counts, actual/static cosine and descriptor difference without
filtering on labels or optimizing any threshold. Models/old readouts are not run.

## Tests and resource gates

Analytic affine fields prove x/y/time routing and half-cell convention. Constant
fields prove border behavior, out-of-frame retention and source nonmutation.
Point/time permutations are equivariant; malformed/nonfinite/nonfloating or
unaligned shapes fail closed. Real eight-clip routing verifies video/media/time/
fine-grid/point-cache provenance, output hashes and fresh-process reload/recompute.

CPU only, four threads for the official sampler, local cloud cost0; new output,
8GiB disk reserve, 300seconds body admission gate and measured peak process working set.
The script checks elapsed time between clips and at completion; this is not an
external watchdog. A blocking operation may return after the limit, in which
case admission fails and the actual elapsed time remains recorded.
At most~202MB new arrays for both arms plus metadata; source caches remain intact.
No new model download, encoder optimization, new QAT/Recovery, product promotion
or main merge. Admission is numerical/feature integrity only. A downstream
controlled readout still needs genuine condition/scene generalization before use.

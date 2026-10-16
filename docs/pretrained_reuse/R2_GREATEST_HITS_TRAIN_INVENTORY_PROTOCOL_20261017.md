# Greatest Hits TRAIN inventory and prospective development partition

Scope: extend the successful fixed TRAIN probe into actual annotation inventory,
then establish task support and a bounded paired-media admission plan. This serves
the conditional AV gate comparison; no model, optimizer or quality scores are used
in this inventory. All previous artifacts and official TEST payloads remain closed.

## Partition fixed before the remaining annotations

Use the pinned official733 TRAIN /244 TEST ID lists and pinned977-parent directory
from the physical probe. A timestamp prefix is the first six numeric ID components;
a date prefix is the first three. Their actual session/actor/scene meaning is not
known. Never assert prefix overlap is proven duplicate media.

Quarantine all324 TRAIN IDs whose timestamp prefix also appears in official TEST.
Retain them in the full annotation inventory and report their label support, but
do not put them in the development training/evaluation corpus. No TEST labels or
media are read. The remaining409 IDs cover26 dates. Sort dates by
SHA256(`17:` + date); assign the first5 dates to development validation and all
remaining dates to training. Freeze the full ID lists and hashes before fetching
new annotations. The resulting101 validation /308 training parents are not a new
sealed audit. Previously inspected eligible probe IDs remain in training.

Validation dates fixed now:2015-10-06,2015-03-25,2015-02-16,2015-09-28,2015-03-13.
No support balancing, seed search, category pruning or date reassignment after
reading labels. Report all missing/unsupported labels. A failed feasibility result
does not justify silently relaxing these rules.

This conservative date grouping prevents known filename-prefix siblings from
crossing development partitions, but does not prove different sites/actors across
dates. Any eventual research fit must disclose this limitation and check actual
asset/near-duplicate identity; it cannot establish the final product quality gate.

## Bounded acquisition and audit

Reuse the three previously acquired TRAIN times files only after matching hashes,
CRC and pinned directory sizes. Acquire each other official TRAIN `_times.txt`
using the existing exact ETag/HTTP206/Content-Range/CRC contract. Never fetch any
other member in this step. Use at most8 independent workers, each with private
range/evidence paths; preflight the aggregate worst-case transfer against64MiB.
Per-request25s/process30s caps remain. Overall dispatch deadline900s; first
failure stops new dispatch, allows in-flight bounded requests to finish and saves
the partial failure. No automatic retry or existing-output overwrite.

Parse all rows as timestamp plus three raw categorical fields; preserve missing
values and unknown classes. Report per development split, date, prefix and parent:
row counts, missingness, material/action/reaction value counts, label support,
option-count feasibility and per-parent class changes. Values are associated with
those semantic fields by the paper and observed tokens; retain the headerless
schema limitation. All acquisition results must be accounted for, including failure.

Native material/action/reaction questions are candidate tasks. They are not assumed
to require both modalities. Before a gate comparison, preserve audio-only,
video-only and question-only controls; define valid intervention labels and measure
joint benefit. Missing annotations never become negative answers. Arbitrary media
mismatches are not automatically labeled incorrect semantic evidence.

Test partition determinism/group disjointness, quarantine, annotation accounting,
unknown/missing handling and transfer selection before execution. Save commands,
source revisions, hashes, wall time and local environment. Then inspect actual
results before proposing/fetching a fixed paired-media subset. No cloud costs,
encoder updates, final evaluation or product promotion.

# Greatest Hits TRAIN group-support audit — 2026-11-28

## Result

A reproducible, TRAIN-only re-analysis of the pinned Greatest Hits annotation
inventory counted each label's support across distinct parent recordings. The
13,327 event rows cover 308 TRAIN parents, but only 7,928 material, 7,946 action,
and 7,833 reaction labels are present. Thus each task has about 40% missing
labels. Missing labels remain missing; this audit neither imputes nor filters
them.

| Task | Known labels | Class count | Lowest distinct-parent support | Classes below 10 TRAIN parents |
|---|---:|---:|---|---|
| Material | 7,928 / 13,327 | 17 | Water: 2 | Dirt, grass, gravel, tile, water |
| Action | 7,946 / 13,327 | 2 | Scratch: 293 | None |
| Reaction | 7,833 / 13,327 | 5 | Splash: 8 | Splash |

The 17-way material task has adequate row counts for some common labels but
very sparse independent parent support for several others: water occurs in 109
rows across only 2 TRAIN recordings; grass has 89 rows across 4; gravel has 49
across 3. For reaction, 112 splash labels come from 8 parent recordings. Those
row counts must not be mistaken for independent task coverage. The action task
has much broader group support, with hit in 307 parents and scratch in 293.

## Interpretation

This is evidence that data support differs sharply by task and class. It helps
explain why the broad material and reaction tasks are poor places to infer a
general model ceiling from aggregate accuracy. It does **not** establish that
the label is wrong, visually unobservable, or the cause of any model error; it
does not show how an unseen-parent evaluation would score. The previously
reported validation distribution shift remains a separate concern.

Do not rebalance, remove rare classes, or redefine the native task based on this
audit. Preserve the current task and labels. Before another fit, obtain more
independent TRAIN parents for the sparse material/reaction classes or choose a
source with better source-group support and independently grounded labels.
Action may be used as a scoped development task only under the existing source
and validation restrictions; this does not admit Greatest Hits for a certified
both-required AV benchmark. Its existing disposition remains
`REVIEW` / `NOT_ADMITTED` for general product training.

## Reproduction and isolation

The audit reads only aggregate TRAIN counts, the TRAIN parent ID list, and
per-parent counts for those IDs. It uses the partition roster to select TRAIN
parents, but does not load validation/quarantine labels or media. The official
TEST and project sealed audit remain unopened. No model, feature cache, or GPU
was used, and no source data or previous artifacts were changed.

Inputs are the preserved local inventory at
`C:\CodexArtifacts\pretrained-reuse-r2\greatest-hits-train-inventory-01\`:

- `inventory.json` SHA-256: `3dab5814a6a5e87599f5d67ad435d7f8a81042e426969c004b9eed6e3d74a19d`
- `partition.json` SHA-256: `6c9ac8eb8e12bd5dc0af77c8d162a40c753d5a215d07eb85dc3a69c3973d9f98`
- `result.json` SHA-256: `409a23c86426a38ddca4456f8d8d3704a1b8d250e516eb926fd53d70c3ae1ab6`
- Upstream inventory source commit: `f4130e200ef390bb38a3bf16667e465896751329`
- TRAIN parent ID hash: `232d37d77ccd010482173a219d9b29769601b29f6a22cc60afd7d0c7cc73141b`

Reproduction command:

```powershell
python scripts/audit_greatest_hits_train_group_support.py `
  --inventory C:\CodexArtifacts\pretrained-reuse-r2\greatest-hits-train-inventory-01\inventory.json `
  --partition C:\CodexArtifacts\pretrained-reuse-r2\greatest-hits-train-inventory-01\partition.json `
  --result C:\CodexArtifacts\pretrained-reuse-r2\greatest-hits-train-inventory-01\result.json `
  --output manifests\pretrained_reuse\greatest-hits-train-group-support-20261128.json
```

The machine-readable output records all per-class row and parent counts and
input hashes. Focused tests cover train-only support, explicit missingness,
duplicate group rejection, and aggregate/per-parent reconciliation.

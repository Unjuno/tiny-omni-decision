# Independent condition-relevance data: actual preparation results

## Result and scope

`PASS_FROZEN_CONDITION_RELEVANCE`: generation and separate-process regeneration
completed. All **12 files match byte-for-byte**, including the result manifest.
This is data preparation, with **zero model forwards, optimizer updates or native
assets**. No quality improvement, completed fit or product promotion is claimed.
The prior semantic fit still selects step0 and fails its advancement gate.

Source commit: `8b23cb5749ecb835721ee0ac8a76bc5654d111be`.
Config: `configs/pretrained_reuse/condition_relevance_v1.json`.
The [prospective protocol](R2_CONDITION_RELEVANCE_DATA_PROTOCOL_20261015.md) and
[compact machine-readable receipt](R2_CONDITION_RELEVANCE_DATA_RECEIPT_20261015.json)
record the allocation, exact pins and verification scope.

## Executed commands and outputs

From `C:/CodexWorktrees/pretrained-reuse-omni`, using
`C:/Users/junny/AppData/Local/Programs/Python/Python311/python.exe`:

```powershell
python -m scripts.prepare_condition_relevance --output C:/CodexArtifacts/pretrained-reuse-r2/condition-relevance-corpus-01
python -m scripts.prepare_condition_relevance --output C:/CodexArtifacts/pretrained-reuse-r2/condition-relevance-corpus-replay-01
```

Both exited0. Each directory contains100,304,025 bytes across12 files.
Both `result.json` hashes are
`7593e36769996d07e3654694013a989a8cdcd8a4b58442a29bde4209feb26964`.
The run verified all six prior input pins and ten source/code pins before and
after generation. Existing data, checkpoints and configurations were read-only.

## Actual coverage

| New partition | Unique states | Questions | Purpose |
|---|---:|---:|---|
| Train |512|8,192|Correct-choice and request-relevance labels|
| Calibration |128|2,048|Reserved binding-only calibration|
| Validation |128|2,048|Independent combinations and question wording|
| Transfer |256|16,384|Matched seen/new query × seen/new candidate wording|
| Total |**1,024**|**28,672**|Synthetic structured-state questions|

All40 candidates remain in every row. Four facts are true, ten candidates answer
the requested attribute, and their intersection is the one correct answer.
Three true-but-unrequested alternatives remain; no correct-field candidate mask
or private task metadata enters the public model input.

The prior896-state reservation (512train/128validation/256transfer) was reconstructed
from its pinned allocation recipe. Its observed train/validation prefixes match.
The previous transfer payload and raw state-partitions file were never opened.
Of1492 previously unused states,1024 were allocated and468 remain unused.
All state and normalized-content intersections between new partitions are zero.

The proposed training pool retains8192 numeric and2048 legacy rows exactly and
adds8192 new binding rows:18,432 total, of which10,240 are explicitly reused.
Old3072+5612 validation rows remain byte-identical. All31 previous slices remain,
alongside16 new slices:10,732 validation questions over47 slices.
Across the complete new+reused pools, state counts are2626train,128calibration,
1756validation and256transfer; all six cross-partition intersections are zero.
These are corpus counts, **not consumed training counts**.

The new train states contain all ten values of each attribute. Their realized
counts vary: color43–66, count40–64, shape37–77, texture45–61. No rebalancing or
duplicate generation was used to hide this variation. Detailed counts and
candidate positions are saved in `accounting.json`.

## Integrity and limits

Thirteen focused CPU tests cover label truth/relevance, ambiguous value names,
private-label poisoning, factorial equivalence, grouping, historical reservation,
disk generation, retained bytes, corrupted hashes and split-isolated loading.
The development loader rejects calibration/transfer/audit requests before file
access. Read-only code review found no blocking issue.

This small authored language bank is a controlled test of condition relevance;
it is not open-domain language coverage. Independent combinations are not
independent real-world scenes. Prior observed errors informed the design.
Calibration covers binding only and cannot calibrate numeric, legacy or native
multimodal quality. Transfer construction and hash comparison do not evaluate a
model on it; transfer and sealed audit remain unavailable to iterative fitting.

## Next decision

Freeze one matched comparison of ordinary correct-choice supervision versus
additional request-relevance supervision, with the same initialization, data,
budget and evaluation. Optimizer, loss coefficient, selection rule and native
transfer protocol are still to be fixed. Do not start several loss/seed/template
variants or treat corpus generation as evidence that the new objective works.

## Repository verification

After both corpus and budget implementations: Ruff passes; CPU pytest reports
**1722 passed,1 skipped** in26.80s; all25 CI manifest-validation/audit commands
pass locally. These checks establish implementation consistency, not model quality.

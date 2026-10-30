# Matched relevance comparison: executed TRAIN-only smokes

Both separate-process, two-update smokes completed successfully at source
`8216c042fe998fabb448d946a3f731a9f9d362e5`. This establishes executable training
and preservation of the native scoring path; it is not a validation result.
No full fit or quality improvement is claimed by this report.

[Frozen comparison protocol](R2_CONDITION_RELEVANCE_FIT_PROTOCOL_20261015.md) ·
[Machine-readable measurements and hashes](R2_CONDITION_RELEVANCE_SMOKE_RECEIPT_20261015.json).

## Executed commands

From `C:/CodexWorktrees/pretrained-reuse-omni`, local Python3.11:

```powershell
python -m scripts.train_condition_relevance --arm ce --smoke --output C:/CodexArtifacts/pretrained-reuse-r2/condition-relevance-smoke-ce-01
python -m scripts.train_condition_relevance --arm relevance --smoke --output C:/CodexArtifacts/pretrained-reuse-r2/condition-relevance-smoke-aux-01
```

| Measurement | CE | CE + relevance |
|---|---:|---:|
| Optimizer calls / consumed questions |2 /36|2 /36|
| Total wall seconds |90.535|91.234|
| Mean seconds per update, two observations |2.215|2.322|
| Projected1024-update training minutes, excluding evaluation |37.81|39.63|
| Peak allocated GPU bytes |3,071,888,896|3,071,888,896|
| Auxiliary gradient norm after clip, update1 |0|0.003068|
| Auxiliary gradient norm after clip, update2 |0|0.001908|
| Native forward equivalence |exact|exact|
| TRAIN probe save/reload |exact|exact|
| Frozen/source integrity |pass|pass|

RTX3080Laptop16GB, FP32/eager. Peak allocation is about2.86GiB; reserved memory
is recorded separately. Two updates give a noisy training-time projection, not
a full-run ETA. Main validation and fixed-train evaluation add substantial time.
Initial reference-cache construction is included in total wall time.

## Matched conditions verified

Both arms have identical code/input pins and byte-identical initial core+aux
delta, retention cache and sampler files:

- Initial delta: `2adf82937665c2eea10c23aed348f512b84b514942be96e92bd26675508817f5`.
- Retention cache: `e976c7b22739e0d0297fedf6830af746c331a24633753935f836ce97c46b676f`.
- Sampler file: `4cd0e805fa558372fd0300ceb2af9b088e4979f05a5c40b7175e9e06d1364272`.
- Ordered question IDs: `2b9cee7d8fb9dab5790b96f32b9deef9bdf5b73fae1a81fae42aa108c2593cfd`.

The actual planned sampler covers18,432 unique questions /2626states once across
1024updates; these are planned counts, while each smoke consumed36 questions.
New binding max length69, numeric147: no new question truncation. Legacy train
has48/2048 questions above512 tokens, maximum786; their state-side truncation
and exact IDs are recorded and identical in both arms. No legacy row was removed.
The tokenizer warning during length inspection is retained in console output;
actual forwards apply the fixed state-only truncation policy.

Both smokes count9452 candidate-pair forwards, including the80-pair native
equivalence probe, reference cache, longest-question backward, updates and
reload probe. The auxiliary layer has385 training-only parameters; it changes
neither the supplied candidates nor the native entailment-logit scoring rule.

No validation, calibration, transfer or sealed-audit payload was loaded by the
smokes. A separate CPU check of permitted development data confirms the complete
10732-row /47-slice main validation schema. Source and existing artifacts remain
unchanged. Full-run initializations will be fresh; smoke deltas are not reused.

## Verification and next execution

Ruff and1730 CPU tests pass (1 skipped). Source-commit push/PR CI runs1784/1785
are successful. Read-only review found no loss/split/selector/checkpoint blocker;
the missing80-pair diagnostic count was fixed before both smokes.

Run CE first, then relevance sequentially, into new directories
`condition-relevance-fit-ce-01` and `condition-relevance-fit-aux-01`, dropping
`--smoke` from the commands above. Preserve the fixed recipe and source during
each run. These are next commands, not a claim that either full fit has executed.
Compare fixed endpoints and independently selected checkpoints only after both
finish and their same-condition checks pass. The existing selected0, all prior
failures and unmet native four-modality quality gates remain unchanged.

# Positive-only auxiliary query gate and acquisition recovery

The128-clip first acquisition terminated with `range length mismatch` after six
verified files. Preserve `fsd50k-training-128-01/result.json` and its completed
media. A second acquisition was launched with identical selection under
`C:/CodexArtifacts/pretrained-reuse-r2/fsd50k-training-128-02/`.

Reuse requires a terminal PASS/FAIL result, exact selection equality, SHA256
verification and then CRC/size agreement against the original ZIP central index.
Old files are copied into the new output, never overwritten. The existing range
reader permits at most two retries for short bodies; exact206/Content-Range and
length checks remain active. No size/hash check was weakened to allow recovery.
Do not treat this launched acquisition as complete until terminal evidence exists.

```powershell
python -u -m scripts.fetch_fsd50k_training_smoke --per-event 16 --reuse C:/CodexArtifacts/pretrained-reuse-r2/fsd50k-training-128-01 --output C:/CodexArtifacts/pretrained-reuse-r2/fsd50k-training-128-02
```

Added `positive_queries(selection)` for the separately fixed auxiliary cohort:
one query per whole clip, only its declared positively annotated event, target
yes. All other events are omitted; no default negative vector is created.
Reject unknown licenses, unsupported/missing positive labels, invalid IDs and
repeated clips/uploaders. Official-train membership/media integrity must still
be verified by the acquisition layer; label mapping is not evidence that the
two datasets' annotation policies are acoustically equivalent.

Frozen128-query SHA256:
`d43c2670d4eea23064a80d873a65ccbe0428cb957ca8a1d80d212fb9000559dd`.
Saved as `positive_queries.json` in the second output. This is a preparation
artifact, not training or a quality measurement. Seven targeted tests and Ruff
pass. No validation records, model weights or previous artifacts changed.

# Native mask-memory comparison: disk interruption, not a quality result

The separately preregistered [4h protocol](R2_NATIVE_MASK_MEMORY_4H_PROTOCOL_20261014.md)
started with all8 known TRAIN videos/696 seeds/two arms/32 selected times unchanged.
The original [2h failed admission](R2_NATIVE_MASK_MEMORY_SMOKE_RESULTS_20261014.md)
is retained. No new training, source annotations, sealed audit or legacy changes.

## Actual execution and saved progress

Source commit `b0d547e3f5edcc90b93e244a986ada3e69e666a9`.

```powershell
python -m scripts.probe_native_mask_memory_4h --output C:/CodexArtifacts/pretrained-reuse-r2/native-mask-memory-full-4h-02
```

PID12944, start12:42:44.167UTC/end13:42:30.117UTC on2026-10-14
(21:42:44–22:42:30JST), body3,585.9457s. Terminal `FAIL_INCOMPLETE`:
`OSError: 8GiB disk reserve exhausted` in native per-frame resource guard.
The process exits; subsequent OS process lookup is empty and GPU utilization0%.
This is an incomplete environment/resource experiment, not rejection of native
memory quality. No automatic restart, partial quality scoring or cohort shrinking.

| Video | Actual complete seeds/frames | Repeat complete seeds/frames |
|---|---:|---:|
|10103|55 /1,760|55 /1,760|
|10186|186 /5,952|186 /5,952|
|10212|43 /1,376|43 /1,376|
|10387|partial49 /1,568; next seed50|not run|
|1087/10886/10894/10903|not run|not run|

Six complete arm archives contain18,176 object-frames. A separately saved
`incomplete-actual-cache32.npz` contains1,568 additional complete object-frames;
not the in-progress50th seed. Planned total44,544. The partial archive lacks
stored seed/frame IDs; current-plan/progress identify its context, not independent
archive identity. It cannot be silently treated as an admitted complete arm.

Output directory totals1,603,135bytes before the separate diagnostic. Free disk
at diagnostic02 is recorded in the machine report. The larger system-wide loss
of free space is **UNKNOWN**; do not attribute it to this1.6MB result directory
or the80.9MB CPU detector download without evidence. No user data/cache deletion.

## Read-only independent product check

[Machine report](R2_NATIVE_MASK_MEMORY_INTERRUPTION_20261014.json), byte-identical
to external `C:/CodexArtifacts/pretrained-reuse-r2/native-mask-memory-interruption-audit-02.json`.
Diagnostic01 also retained;02 follows formatting of the diagnostic source only,
with no changes to inference or old outputs.

```powershell
python -m scripts.record_native_mask_memory_interruption --main C:/CodexArtifacts/pretrained-reuse-r2/native-mask-memory-full-4h-02 --output C:/CodexArtifacts/pretrained-reuse-r2/native-mask-memory-interruption-audit-02.json
```

All50 pinned media/model/library/protocol/code inputs remain byte-identical.
Actual complete archives match recorded SHA256/size/schema and planned seed/frame
IDs. All19,744 saved mask rasters independently reproduce bbox/empty flags;
scores finite and raw stream digests have the expected format. Raw full logits
are not reconstructed. Partial identity limitations are explicit.

Status `PASS_SAVED_PARTIAL_GEOMETRY_INTEGRITY_ONLY` describes this CPU check.
Final model-state hash, RNG/freeze and peak-resource assertions were not reached
on exception and are **UNKNOWN**, not success. Full-main admission correctly
rejects the terminal failure before fresh replay or proxy annotation access.
The complete auditor/scorer and barrier tests are implemented but not executed
against this incomplete run. Four new interruption regression tests cover shape/
geometry corruption, missing identities and ordered cohort-prefix accounting.

## Unperformed work and next conditions

Full inference, fresh2/696-seed numerical replay, full product admission and
posthoc24-group proxy comparison remain unperformed. No group quality,
Accuracy/NLL/Brier/ECE, task adoption or95–98% Omni claim follows from the prefix.

The existing main needs8GiB reserve plus450MiB output admission budget;15GiB
free is requested to leave practical headroom. Capacity alone is not an
instruction to auto-restart. A continuation must separately specify new output,
honest source commit/timing, unchanged full cohort, integrity of reusable complete
arms and treatment of incomplete arm/native cache. No exact process/cache resume
is supported by these saved arrays; no hash checks or reserve may be bypassed.
Preserve the old failure and all products. No paid compute, quantization,
main merge or product promotion. The overall Omni goal remains active/unmet.

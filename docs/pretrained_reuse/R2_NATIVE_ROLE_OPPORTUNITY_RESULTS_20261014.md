# R2 training semantic-role opportunity census — executed 2026-10-14

## Finding and denominator

After complete native/derived-feature admission, inspect source training geometry
only. All8knowntrain×32times/all18,441NMS candidates;32 source track instances
(24group+8hidden),1,024 unique(track,time) cases. The8 designated covers are a
subset of24 group instances; cover counters are an overlapping view, never added
as8 extra independent objects. No training/readout/QA predictions/dev/sealed/GPU.

| Source role view | All cases | Box available | Missing | Raw IoU>=0.5 | NMS IoU>=0.5 |
|---|---:|---:|---:|---:|---:|
| Hidden target |256|243|13|44|41|
| Group members |768|768|0|608|589|
| Designated cover (group subset) |256|256|0|201|193|

Hidden and designated cover boxes are simultaneously available243/256times, and
both have an NMS overlap opportunity30times. **25 of41 hidden opportunities and
20 of30 co-support times come from10103 alone.** This strongly concentrates the
available geometry for this particular supervision proxy. It does not prove lack
of pixel evidence, model capacity, semantic failure or a detector/benchmark ceiling.
Do not turn all unmatched candidates into negatives or repeatedly fit the same
few cues and call that unique data coverage. Keep difficult/unavailable cases.

| Known train clip | Hidden box available | Missing | NMS hidden overlap | Same-time hidden+cover overlap |
|---|---:|---:|---:|---:|
| video_10103 | 32 | 0 | 25 | 20 |
| video_10186 | 32 | 0 | 2 | 1 |
| video_10212 | 32 | 0 | 1 | 1 |
| video_10387 | 32 | 0 | 4 | 2 |
| video_1087 | 32 | 0 | 3 | 3 |
| video_10886 | 32 | 0 | 2 | 1 |
| video_10894 | 32 | 0 | 3 | 2 |
| video_10903 | 19 | 13 | 1 | 0 |

These are correlated training frames and **bbox-overlap opportunities**, not
recognition Accuracy or successful cover/memory decisions. Each maximum/best-mask
index is posthoc GT geometry only; it never selected an inference mask or changed
inputs. The relation/memory task can require evidence accumulated across times;
same-time overlap is not assumed to be a necessary condition for answering it.

## Sparse annotation and masked-flag limits

Use existing tested range-limited interpolation, with exact/bracket/nearest-distance
metadata. Hidden available cases15exact/228interpolated; group54exact/714interpolated;
cover18exact/238interpolated. All13 unavailable hidden times belong to10903 and
remain missing, not detections that failed. Annotation start is not physical birth.
`is_masked` brackets are recorded without claiming that they certify visibility:

| Hidden source bracket flags | Available cases | Raw overlap | NMS overlap |
|---|---:|---:|---:|
| [0,0] |213|29|27|
| [0,1] |13|7|7|
| [1,1] |6|4|4|
| [1,0] |11|4|3|

This supplemental flag stratification describes the saved census, not a visibility
assessment or a newly chosen subset to improve quality. No threshold change.
Independent audit recalculates all1,024 cases' raw/NMS IoUs/matching counts with
separate scalar arithmetic and recounts every clip/role/co-support denominator.
MaxIoUerror0.0, unchanged overlap threshold0.5; verification abs1e-12/relative0.
It **does not independently reconstruct source bbox interpolation**, and checks
the source ZIP hash without opening it. Keep this verification limit explicit.

## Actual execution and provenance

Census body6.766895s; independent recount2.894523s, CPUonly.
Input parents/config/banks/fresh/full audits and helper/protocol bytes hash-bound
**before** the first training ZIP read and again at exit. NumPy RNG unchanged.
Original trainZIP SHA
`9eaa6aad3fabdf4cde4955f333cfddf3b9e6839c5abcb271afa410870f640d78`.
The [fixed protocol](R2_NATIVE_ROLE_OPPORTUNITY_PROTOCOL_20261014.md) was committed
before labels: `fbbd0dbbeee83b2beaa41df30b3d77ed0a197927`.
[Companion JSON](R2_NATIVE_ROLE_OPPORTUNITY_RESULTS_20261014.json) retains all
summaries/hashes; local full result saves all1,024 cases and source pins.

```powershell
python -c "import runpy;runpy.run_path('C:/CodexArtifacts/pretrained-reuse-r2/diagnose-native-role-opportunity-01.py',run_name='__main__')"
python C:/CodexArtifacts/pretrained-reuse-r2/audit-native-role-opportunity-01.py
```

Both completed outputs are fresh/write-once. Census producerSHA
`08a75cf2a9071efc136e19318cca4611a48935f03fc217e9fd9114ba23ece536`; independent recountSHA`3ed42bb0edfdd0ff76dff7b1f30de1f038e6149525f4dc094a27ea10befae942`.
Existing tested source-role parsing/geometry helpers reused; no protected model/
run/corpus/config/checkpoint changes. Full CPU suite1,321passed/1skipped, Ruff and
manifest/catalog/policy checks passed before publication; read-only census review
found no issues. Exact publication-head CI follows push, saved locally.

## Consequence for the next quality experiment

The new foreground bank is admitted, but instantaneous hidden/cover box supervision
is sparse and concentrated. Define cue-time selection from inputs and time-aware
identity/memory targets, preserve unknown/ambiguous cases, and establish scene-grouped
controls before fitting. Do not use GT best masks as runtime observations, silently
replace the difficult task, or extend a failed QA readout merely because more
correlated mask rows exist. Audit useful cue/supervision diversity before larger
training. All8 clips have been observed; they cannot be presented as sealed evidence.

No semantic Decision Accuracy/NLL/Brier/ECE is measured here. The<=200M calibrated
JSON/native text/image/audio/video joint/OOD95–98% product remains incomplete.
No quantization, model promotion, main merge, cloud or paid compute in this unit.

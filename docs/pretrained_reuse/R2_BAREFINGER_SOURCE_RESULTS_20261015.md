# Bare-finger contact source — actual native inspection, semantics unresolved

**Full source inventory and native read/replay PASS; training NOT_ADMITTED.**
This is an accessible public image/contact-stream candidate while VisTouch's
native archive remains inaccessible through the current supported route. The
inspected trial/photo archives contain no RGB video and establish no joint
Decision benefit. No model was
trained, inference performed or four-modality95–98%/<=200M goal completed.

## Primary release and actual access

[Publisher release](https://springernature.figshare.com/articles/dataset/26965267),
`doi:10.6084/m9.figshare.26965267.v1`, official immutable version1. Official
API/license URL explicitly grants CC BY4.0 to the dataset. Preserve attribution
to Devillard, Ramasamy, Cheng, Faux and Burdet. The supplied source text was read,
never imported or executed; data terms do not establish a separate code grant.

The source API advertises `datah5.zip`20,384,594,825 bytes; it was **not** fully
downloaded. Exact HTTP ranges read all three ZIP directories in101,139 bytes,
with strict Content-Range/length checks, then one fixed pilot. Member ZIP CRC
and local SHA256 are verified; full-archive MD5 is not verified by partial reads.
The immutable version/README/API/directory/payload receipts remain outside Git.
Directory retrieval took23.546s; fixed payload transfer120.938s,44,950,181 range
body bytes (including directory/local-header reads). No archive account or client.

## Complete cohort accounting

[Prospective protocol](R2_BAREFINGER_SOURCE_PROTOCOL_20261015.md), executable
source commit `8c6cc0c40f8293cae0a5b875f3378a0aece3c805`.

| Source quantity | Actual complete directory |
|---|---:|
| Participants |10: subject_0 through subject_9 |
| Texture identities |10:7t,10t,18t,39t,44t,54t,59t,83t,108t,120t |
| Physical contact trials |500; exactly5 per participant/texture |
| Trial types |std_0/std_1/std_2/cte_force_3/cte_speed_4;100 each |
| Participant/texture groups |100 |
| Stereo surface photos |200;10 shots×2 cameras per texture |
| Shared photograph/specimen groups |10 texture-level groups |
| Native RGB video members |0 in these directories |
| Independent validation |Not defined/admitted |

All500 Cartesian identities and all200 photo names are present; no missing trial,
pruned surface, duplicated filename or new split. The source contains one
texture-keyed `force_offset` JSON, not semantic labels for microphone channels.
Separate participant identities cannot make the shared texture photo independent.
Never reuse the same photograph in train and validation or apply the correct
texture's calibration by knowing the target. Physical specimen generalization,
full media/content overlap and an independent task protocol remain open.

## Fixed pilot: raw values, no processing or semantics repair

One unique source trial `datah5/subject_0/7t/subject_0_7t_std_0.h5`, plus the
surface-level `img/7t/7t_l_0.jpg`, calibration JSON and two source-text files.
No train/evaluation split role was assigned and no additional trial was chosen
from quality results. HDF5 payload44,633,568 bytes, SHA
`67bd49b0f8ea5069045c45c95c2c3217a60d122aba33fa2b37a908288206e5a4`.

The [paper](https://www.nature.com/articles/s41597-025-04670-0) describes two
microphones, two accelerometers and force/position streams with timestamps.
However its `kistler_data` ordering lists vibrations before audio; the released
`HAVdb/importer.py:get_kistler_features` labels columns0:2 audio and2:4 vibration.
The actual HDF5 datasets have **no attributes or column labels**. Thus audio/
vibration assignment is `UNVERIFIED_CONFLICT`, not resolved by guesses about the
signals or by whichever channel yields higher Accuracy.

| Native value array | Shape | First/last supplied timestamp | Span(s) |
|---|---:|---|---:|
|kistler_data|759,808×4|918.606400013 /979.391071081|60.784671 |
|ft_sensor_data|745,099×6|918.702165842 /979.552292585|60.850127 |
|pos_data|1,826×4|918.688005924 /979.527781963|60.839776 |

All six value/time arrays are float64; timestamps are N×1. All values are finite,
all timestamps strictly increase in this pilot, with zero duplicates/reversals.
Timestamp gaps vary (maximum force gap0.0171118s, position0.0605748s); these are
observed data intervals, not proof of clock precision or missing data. Position
has4 actual columns versus8 in the paper's schema prose. Supplied times are
around918–980, not absolute Unix epochs. The source loader applies a fixed
negative epoch offset, which this probe **does not apply**. Export-time basis,
fine cross-sensor accuracy, physical units and calibration remain unverified.

Only six local hard-linked numeric datasets and built-in compression filters
are accepted. External/soft links, virtual/external storage and unknown filters
are rejected; plugin search paths are empty.72,190,904 raw array bytes inspected.
The JPEG actually decodes to3264×3264 RGB; it is a static photo of texture7t,
not a time-matched finger/contact video. Visual inspection of this one photo
shows no visible text annotation; other199 pictures were not fetched/inspected.
No resampling, gain changes, interpolation, label-dependent offset or invented
clock was used; original source bytes remain unchanged.

## Execution and fresh readback

```powershell
python -m scripts.audit_barefinger_source --source-root C:/CodexArtifacts/pretrained-reuse-r2/barefinger-source-directory-01 --source-result-sha256 68310284d4b9e660267d76301e5da1cc485e43d74c9ca1b43a5a6cd51bc7ed19 --output C:/CodexArtifacts/pretrained-reuse-r2/barefinger-full-source-audit-01.json
python C:/CodexArtifacts/pretrained-reuse-r2/barefinger-native-probe-01.py
python C:/CodexArtifacts/pretrained-reuse-r2/barefinger-native-replay-01.py
```

| Execution | Seconds | Peak CPU bytes | Evidence |
|---|---:|---:|---|
| Full metadata audit |0.168063|44,376,064|13 bound sources/code/protocol |
| Complete fixed numeric/image inspection |1.634506|211,963,904|733 pins |
| Fresh separate-process native replay |0.931146|206,651,392|735 pins; all7 full arrays byte-exact |

The replay validates all six full HDF5 arrays and full RGB pixels after reopening
the native files. It shares HDF5/Pillow and does not independently establish
audio semantics, sensor ground truth or synchronization. No model Save/Reload
or quality result is claimed. All raw-array hashes and summaries are retained.

HDF5 was absent from both current and bundled Python. A fixed h5py3.15.1 binary
wheel was installed with `--no-deps --target` in a **new isolated artifact path**;
existing NumPy2.4.6 and all legacy environments stay unchanged. Actual HDF51.14.6;
installed files, DLLs, NumPy/Pillow implementations and source/pilot receipts are
hashed. The pip report records the wheel's published download SHA; the wheel
archive itself was not retained as a separately verified local artifact. No
dependency upgrade, GPU execution, paid/cloud compute or legacy restart.
Electricity cost was not measured. Transfer-time CPU peak was not measured.

Full accounting SHA `dea100ea10b16a799108015ea58e1e147aec18d4249b23529e33aa5d3aa40a3d`;
native inspection `8fc66000b7771c5dbbd2cbbd1c9dfb81cda602ea904894c5e92626bc89cd3c65`;
fresh replay `33f3dded65a1e0cdc4f720b01ae7c4426f570665b0fcca988e1b25544bfda99f`.
[Small exact summaries](R2_BAREFINGER_SOURCE_RESULTS_20261015.json) include source,
payload/hash/accounting records. Large HDF5/JPEG/runtime files and the fixed
native producers remain under `C:/CodexArtifacts/pretrained-reuse-r2`, outside Git.

## Next action, without changing the product goal

Do not admit this source as typed Audio+Video from filenames or collection claims.
Resolve audio/vibration ordering, physical units and label-independent calibration
from traceable primary evidence. Define an independent participant/specimen/task
protocol which avoids reusing photos, and retain all textures/unique trials.
Use genuinely video-bearing sources separately; position traces are not video.
Only then compare frozen native singleton/joint inputs with no-input, mismatch
and source-length controls. Existing negative experiments and the interrupted
GPU run remain protected. This source probe adds zero product parameters and
does not establish generic condition following, OOD, calibration or95–98% quality.

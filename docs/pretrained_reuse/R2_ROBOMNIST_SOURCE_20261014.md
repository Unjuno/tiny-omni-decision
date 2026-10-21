# Public joint-input candidate: RoboMNIST source feasibility

User authorized selecting public data for synchronized audio/video/optional-sensor
decisions. The full <=200M prequantization native text/image/audio/video calibrated
Decision/JSON and95–98% task/OOD/joint quality goal remains unchanged and unmet.
This is a new source feasibility check, not another fit on failed VIDIMU/AV
correspondence/occlusion selector sets.

## Official release actually checked

[Official dataset version1](https://doi.org/10.6084/m9.figshare.28179383.v1),
[versioned API](https://api.figshare.com/v2/articles/28179383/versions/1),
[pinned author README](https://github.com/SiamiLab/RoboMNIST/blob/70f5e6393acf6128a7ceb7afacf4463060626ca3/README.md).
Official dataset metadata grants CC BY4.0. Released data rights are ALLOW with
attribution; training admission is still gated by content, split and timing checks.
ActionSense was also screened using its [official license](https://action-net.csail.mit.edu/index.html):
CC BY-NC-SA4.0, so it is not selected for this product path; no media acquired.

RoboMNIST records two robot arms, digit-like trajectories and three velocity
levels, with synchronized video/audio/Wi-Fi CSI. Prospective typed decisions:
which robot moved, which trajectory, which speed. These are narrow controlled-lab
tasks; useful joint signals and generalization must be measured, not presumed.
Labels/source filenames/robot position ground truth never become runtime inputs.
The full goal is not reduced to these tasks.

Actual version metadata:60 archives totaling56,403,232,879bytes (52.53GiB).
All60 class indices and robot×velocity×action combinations are present.
`31_Ar2_Vmed_Ac1.zip` has no populated computed MD5; this is retained as a source
integrity limitation, not discarded. Whole archives were not downloaded/hashed.

## Actual bounded ZIP-directory inspection

Fixed first archive `00_Ar1_Vhigh_Ac0.zip` (889,834,555bytes); exact HTTP206
Content-Range/response-length checks, TLS intact, transfer cap524,288bytes.
Only75,450 directory bytes transferred. No member payload opened/extracted,
model inference, source-label scoring or training.

Archive has546 entries including directory and macOS resource forks.
Exclude273 resource forks from data inventory, retaining272 actual files:
102 video,102 audio,34 CSI and34 robot-position files.
These form **34** complete repetition/uncertainty groups, not272 independent
examples. Three camera/mic views and the CSI from one repetition share one split.
Position JSON is supervision-only. This one directory has34 repetitions; the
authors' minimum32 must not be extrapolated into a complete-corpus observed count.
Other59 group inventories, payload synchronization and session independence remain
UNKNOWN. Do not claim participant/lab-disjoint OOD from one laboratory/two arms.

[Machine receipt](R2_ROBOMNIST_SOURCE_20261014.json) stores metadata/code/source
hashes. Full directory and34-group stream mapping remain outside Git under
`C:/CodexArtifacts/pretrained-reuse-r2/robomnist-source-audit-01/`.
Eight pure CPU regression cases cover all-stream grouping, resource forks,
folder/file disagreement, unsafe paths, missing/duplicate streams and exact
archive-to-member class binding. Review found that internally consistent members
from another class could be misattributed to the inspected ZIP; the explicit
archive binding now rejects them. Original result01 remains outside Git and the
published report is the rechecked result-verified-02; all34 actual groups unchanged.

```powershell
python -m scripts.audit_robomnist_source --output C:/CodexArtifacts/pretrained-reuse-r2/robomnist-source-audit-01/result-verified-02.json
```

## Next admission before any learning

Acquire directory inventory for all60 archives with a fixed bandwidth/time cap;
pin complete physical-repetition identities and reject overlap across splits.
Then independently inspect one preregistered complete repetition's permitted
payload timestamps/schema/decode/checksums, with source filenames hidden from
model inputs and all receivers grouped. Do not download52.53GiB to begin this work.
Freeze task/options/splits and question-only, audio-only, video-only, CSI-only,
joint and evidence-shuffle controls before fitting. Compare calibration and
paired uncertainty at repetition level. Same-lab split limitations remain explicit.

No new readout, weights, fitted probabilities or training corpus is produced here.
The capacity request for the interrupted native-memory diagnostic is pending;
source metadata work proceeds independently with existing artifacts protected.

# Complete public RoboMNIST directory inventory actually executed

This continues the [source feasibility](R2_ROBOMNIST_SOURCE_20261014.md) with the
[fixed directory-only protocol](R2_ROBOMNIST_DIRECTORY_PROTOCOL_20261014.md).
The full <=200M prequantization native text/image/audio/video calibrated
Decision/JSON and95–98% task/joint/OOD product goal remains open. No goal narrowing,
fit, model inference, new GPU run, source payload/GT or sealed audit access.

## Actual run and independently rebuilt products

Producer source commit `765c4595730230582f62cc5bff0625278f1802d5`; all seven source/
protocol/test/version-metadata pins unchanged. Existing Python3.11 environment,
Windows CPU/network only; no dependency upgrade/cloud. Process PID54120 was
observed with exact command/start while live, then its tool session exits0.
Start/end2026-10-14T14:23:59.523UTC /14:30:40.688UTC
(23:23:59–23:30:40JST).

```powershell
python -m scripts.audit_robomnist_all_directories --output C:/CodexArtifacts/pretrained-reuse-r2/robomnist-all-directories-01
python -m scripts.replay_robomnist_directories --main C:/CodexArtifacts/pretrained-reuse-r2/robomnist-all-directories-01 --output C:/CodexArtifacts/pretrained-reuse-r2/robomnist-all-directories-replay-02.json
```

The producer's exact source snapshot/config/hashes, per-directory members/groups,
60 progress receipts and terminal result are external. CPU replay independently
reconstructs identities using a separate lexical implementation, verifies source
archive/root/class consistency, actual file SHA256/config/pins/cohort, all input
streams and supervision-only positions, counts and transfer limits. It is saved
metadata validation, not independent network reacquisition or payload verification.
The replay source was uncommitted during execution; exact auditor SHA256 is recorded
in the report and its source/tests are published alongside this record. The
report's `source_commit` refers to the producer, not that uncommitted auditor.

[Machine report](R2_ROBOMNIST_DIRECTORY_RESULTS_20261014.json) ·
[Every class/count](R2_ROBOMNIST_DIRECTORY_COUNTS_20261014.csv).

| Actual measurement | Value | Predeclared bound |
|---|---:|---:|
| Complete ordered class archives |60|all60|
| Main body seconds |401.1661|900|
| Application response body bytes |4,691,716|per archive524,288|
| Reserved transfer bytes, including failed-read reservations/length-check allowance |4,691,896|12MiB|
| Complete output files, including terminal/source/config/progress |7,385,225bytes|16MiB|
| Independent CPU replay seconds |0.14|small metadata-only check|
| Unique repetition/uncertainty groups |2,135|not assumed32/class|
| Real source member files |17,080|8/group|
| Resource forks excluded from media inventory |17,140|not observations|

All60 classes have32–44 repetitions. Video6,405 files, audio6,405, CSI2,135,
supervision-only robot position2,135. All3 camera +3 microphone +1 CSI streams
share their physical repetition group; they are not independent examples across
splits. No difficult group/class or missing-MD5 source is removed. Actual counts
supersede extrapolating the paper's minimum32 or the first archive's34 to all data.
One archive still lacks a populated computed MD5 as reported by the prior source
audit; whole-archive MD5 and member CRC/payload SHA are unverified.

The fixed first archive agrees with the earlier34-group source receipt. A
read-only scan of input-member CRC32+uncompressed-size metadata found zero
cross-group repeated signatures. This weak directory hint does **not** prove
zero byte/decoded/normalized content overlap or collection-session independence;
actual payload SHA/content checks remain mandatory before dataset admission.
The external `robomnist-directory-signature-observation-01.json` receipt covers
all14,945 input members and binds replay02; SHA256
`31547cf150841bea787fe4e52d8ea26d613f21a17cfa5faef36a900101546e88`.

## Review correction and resource context

Replay01 retained. Review identified that individually bounded row transfers and
a bounded reported final reservation were not enough: an understated cumulative
reservation could pass. Replay02 binds positive/integer per-archive bytes,
monotone/sufficient cumulative reservation increments, actual cumulative totals,
last-row equality and final12MiB cap. Corruption regression RED→GREEN; actual
2,135 groups and17,080 files unchanged. Both producer/replay source regression
tests reject partial cohorts, class substitution, missing streams and positionGT
input contamination. No remaining Critical/Important review finding.

The separate native-memory GPU experiment remains disk-interrupted with its8GiB
guard unchanged. This tiny metadata task's1GiB reserve is a separate protocol;
it did not restart or alter that experiment, a Teacher, a frozen corpus or weights.
No cost beyond existing local CPU/network; paid compute0. RAM/CPU peak and physical
network/TLS overhead bytes were not measured, so application transfer counts must
not be presented as those measurements.

## Remaining admission and quality gates

Inventory PASS establishes complete directory/group bookkeeping only.
Actual timestamps, waveform/video decode, CSI schema, near-duplicate content and
collection-session grouping remain unverified. No train/validation/audit corpus
or probabilities are produced. One controlled lab/two robots does not establish
household/participant/population OOD.

Next inspect a preregistered complete repetition's permitted input payloads and
timestamps in a separate bounded artifact; keep positionGT and source filenames
outside runtime inputs. Then freeze task/options/source-group splits and audio-only,
video-only, optional-CSI-only, question-only, joint and evidence-shuffle controls.
Do not assume fusion benefit or95–98% from a public source's existence, and do not
repeat the previously closed readout/seed/prompt fits to create a favorable score.

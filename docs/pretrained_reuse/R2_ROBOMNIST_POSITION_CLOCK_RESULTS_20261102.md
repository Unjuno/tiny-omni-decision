# RoboMNIST PositionGT clock audit — 2026-11-02

## Outcome

**Clock bridge: UNVERIFIED. Stop this source path before model fitting.**

The one permitted PositionGT member was fetched from the already observed
TRAIN-side group `Ar1_Vhigh_Ac0_Rp01_UNC062`. Its ZIP member size, compression,
CRC32, and SHA-256 matched the pinned metadata. The bounded download transferred
107,525 bytes total; the extracted member was 313,307 bytes, CRC32
`082e8e4d`, SHA-256
`572c20c11cf16d2424b624bba6ae49a0652c060372bb3ccce126e9ed501bb176`.

The timestamp-only schema scan found no timestamp/clock/epoch-like field names,
so it yielded no PositionGT timestamps to compare with the previously observed
CSI timestamps or video PTS. The upstream README says the central monitor
synchronizes collection and that camera/CSI/audio nodes timestamp their data,
but it does not specify a shared clock/epoch, units, or a cross-stream mapping.
That general description is insufficient to establish alignment within one
video frame period. This result does not establish WAV/audio alignment either.

The raw PositionGT JSON was parsed only in memory and was not saved. Position
and label values were not reported or used. This group remains training-side
only; no model, fitting, GPU, sealed audit, or cloud compute was used.

## Timing and resource record

- Source commit: `9ff2e320`; runner SHA-256:
  `4196be2b8e966033b5bef4d1db17c05289c4735603247d8d48ff0c9e0800c574`.
- Protocol SHA-256:
  `477c1d9f759c56856fceef0c2f9d58264d1376a75274a8876b283493ef9e8445`.
- Effective config SHA-256:
  `63b485b7229ed9dee7e615bc3867b4713e0fb40fa9e201640d294fcb15e69ece`.
- UTC start: `2026-11-02T06:22:28.616013+00:00`; elapsed: 14.229 seconds.
- Seven exact HTTP range reads averaged 2.024 seconds each. The first four
  averaged 2.073 seconds; after observing them, the controller waited 3 seconds
  (mean plus ~1 second margin) before its next status poll. No retry occurred.
- Peak working set: 38,817,792 bytes. GPU used: no. Cloud cost: 0.
- Output directory:
  `C:\\CodexArtifacts\\pretrained-reuse-r2\\robomnist-position-clock-audit-20261102-attempt-01`.
- Audit JSON SHA-256:
  `82b0c8b78e0241f9d81ad762bf120ef83864bb378ac2b97a54d238eb5a61d36f`.

The download itself passed integrity checks, but the experiment does **not**
pass the clock-bridge gate. Do not treat this one group as multimodal
generalization evidence, and do not start a video+CSI fit from it. A future
attempt would need explicit timestamp-bearing PositionGT schema or authoritative
documentation describing a common acquisition clock and timestamp mapping;
the original member was not retained, so this audit must not be described as a
reproducible retained-data experiment.

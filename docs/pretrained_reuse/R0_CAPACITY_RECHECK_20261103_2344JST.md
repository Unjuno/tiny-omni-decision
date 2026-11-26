# R0 capacity recheck — 2026-11-03 23:44 JST

The current branch is `docs/pretrained-reuse-roadmap-20261010` at
`eb7f09787d1b012013416a6ab4e6c043ea37b933`, matching origin. The only
untracked repository files are the two user-owned PhysicalAI split manifests;
they were not changed or staged.

The active Quartus processes (`quartus_sh` PID 12696 and `quartus_fit` PID
26024) exited during observation. Three C: samples taken after they were gone
were:

| Local time | C: free bytes |
|---|---:|
| 23:44:10 | 1,568,718,848 |
| 23:44:20 | 1,568,628,736 |
| 23:44:30 | 1,564,430,336 |
| Mean | 1,567,259,307 |

The mean is **1.460 GiB**, with a 4.09 MiB spread, still **580,224,341 bytes
below** the feature extractor's fixed 2 GiB reserve. This shows that waiting
for this Quartus run to end did not restore enough capacity. The RTX 3080 was
idle at the initial recheck; no training or extraction process was active.

The RoboMNIST AV-match feature cache remains `INCOMPLETE` at 1,778 / 1,780
groups. The absent entries are still the two frozen validation groups; no
partial validation evaluation was run. The train-only checkpoint and the
original feature cache were not modified. The cache extractor, split, 2 GiB
guard, and both user-owned manifests remain unchanged. No artifacts were
deleted in this recheck.

Resume the exact existing cache only after C: has stable headroom above the
2 GiB reserve. The extraction is CPU-only; GPU availability is not the current
limiting factor.

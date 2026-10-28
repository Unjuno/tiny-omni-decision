# MEVA annotation repository acquisition record (2026-10-28)

## Decision

This was a bounded source-metadata acquisition, not an admitted training run.
The official annotation repository is pinned at
`421841a75577b697c314e952e585aecbb1b99e17`. Its README distinguishes
`kitware-meva-training` annotations from `kitware` evaluation-level
annotations, and names videos as five-minute clip IDs containing date,
time, site, and camera. The root `LICENSE` declares CC BY 4.0 for the
annotation repository. These facts support a possible training-only,
clip-grouped video task, but do not by themselves establish rights to every
underlying video asset or define a clean local validation split.

## Acquisition and disk evidence

Command:

```text
git clone --filter=blob:none --depth 1 https://gitlab.kitware.com/meva/meva-data-repo.git C:\CodexArtifacts\pretrained-reuse-r2\meva-metadata-screen-20261028-attempt-01
```

The clone process began at 2026-10-28 23:03:42 JST. It was stopped by the
operator after a later check showed only 1.50 GiB free on C:, with the drive
subsequently observed at 0.82 GiB free. The partial directory was preserved.
At the read-only inspection it contained 14,329 files totaling
5,955,001,894 bytes; the Git tree names 16,763 paths and `git status` reported
16,763 tracked entries, so this is not a completed, clean checkout. The tree
is annotation/documentation/metadata; no MEVA video clips were acquired.

The machine has only C: available to this task, so no feature cache, model
checkpoint, or training output should be created there until the user frees
space. Do not resume this clone or download video media under the current
capacity. Do not delete this partial acquisition automatically.

## Transfer-rate observations

Three prior 1 MiB ranged samples from OpenSLR measured 0.324, 0.3828, and
1.770 MiB/s (arithmetic mean 0.8256 MiB/s). Their spread is large and they
are not MEVA-host measurements, so they do not support a reliable MEVA ETA.
The separate ADL archive acquisition's first-three-file average of
0.2136 MiB/s is also source-specific. A future ETA requires three repeated
measurements against the actual approved payload and network endpoint,
including each file's completed bytes and elapsed time; do not combine these
different sources into one mean. Command sleeps can reduce polling overhead,
but must be followed by checking the same live transfer handle and available
disk before continuing.

## Runtime snapshot and next gate

At 2026-10-28 23:12 JST, the RTX 3080 Laptop reported 15,821 MiB used and
100% utilization. Orbitune training PIDs 31624 and 22136 were active; neither
was modified. No new model training was started. A MEVA fit is not admitted
until there is storage headroom, video-asset rights are verified for the
specific subset, and a training/validation split is fixed by independent
video identity without using the evaluation-level annotations.

## Official dataset-license follow-up — 2026-10-29

The official [MEVA data page](https://mevadata.org/) states that all MEVA data
is under CC BY 4.0. The linked [dataset license](https://mevadata.org/resources/MEVA-data-license.txt)
identifies the covered material as the MEVA dataset by Kitware and IARPA. The
official WACV paper also describes the release as CC BY 4.0 and states that
actors signed consent forms under IRB oversight. This resolves the earlier
source-level uncertainty about whether the public MEVA video release is
licensed, but not all rights: CC BY grants only rights the licensor can grant,
and its text expressly leaves privacy/publicity rights outside the copyright
license. Preserve attribution and source provenance.

This makes a bounded **video-only** development task a plausible candidate;
MEVA is not synchronized natural audio/video or video/IMU evidence, and no
two-input necessity claim follows. The partial annotation checkout remains
incomplete and preserved. No MEVA video was acquired. Before training, still
pin an exact public training subset and annotation revision, group every
view/clip from the same underlying event or actor appropriately, construct a
validation split without evaluation-level annotations, audit label/task
ambiguity, and pass the 2-GiB output-storage and GPU gates. The machine-readable
license recheck is
[`meva-source-license-recheck-20261029.json`](../../manifests/pretrained_reuse/meva-source-license-recheck-20261029.json).

## Training-annotation subtree audit — 2026-10-29

Read-only inspection used only
`annotation/DIVA-phase-2/MEVA/kitware-meva-training` at the pinned commit; no
annotation content from the separate `kitware` evaluation-level directory was
read. The local training subtree exactly matches all 4,331 Git-tree entries
at the pinned revision. It contains 1,443 `*.activities.yml` clip files;
`status.csv` and an independent YAML parse agree on 35,174 activity instances
across 37 labels. The directory-level README describes these as training-level
annotations.

The 1,443 camera clips map to 741 distinct exact `(date, start, end, site)`
windows across four sites and seven dates. A conservative `(date, site)` split
has only 28 groups. These are scene/time groups, not confirmed independent
people: stable subject identity across different clips is not established.
Class counts are extremely long-tailed: `person_abandons_package` has 1
instance and `person_steals_object` has 3; the largest class has 3,961. Thus a
high-accuracy 37-way result would be difficult to validate with a clean
participant-disjoint split. Do not prune rare classes after seeing validation
results; any narrower task needs a prospectively justified operational scope
and its own untouched split.

The exact subtree check found 4,331 expected and 4,331 present files, with no
missing or extra paths. Canonical SHA-256 over the sorted `git ls-tree -r`
entry lines is `e9c75e2343f29137dea04766ac8d72e3c6511f06d4e29fdc6029337ec288a638`.
`README.md` SHA-256 is
`31d9917b93dce3747ae6a3db5bbc28ef36805c2eec4f1bce8e9a27fb8a0eeadd` and
`status.csv` SHA-256 is
`8c7578e8d65d352ea9ac4254ef97752ec10bce47798cfac99b030a934bbfa9d4`.
See [`meva-training-annotation-preflight-20261029.json`](../../manifests/pretrained_reuse/meva-training-annotation-preflight-20261029.json)
for all class counts and the group inventory. This resolves metadata availability
for training annotations, not video access, a clean subject split, or runtime
feasibility. No video was acquired and no fit was run.

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

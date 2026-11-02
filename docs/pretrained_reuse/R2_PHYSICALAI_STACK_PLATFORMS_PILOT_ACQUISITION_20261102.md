# PhysicalAI stack-platforms pilot payload acquisition — 2026-11-02

## Disposition

This records a bounded, pinned train/validation payload acquisition after the
metadata-only screen. It is not a fit, inference run, quality result, or source
admission. `panda-stack-platforms` remains `REVIEW_NOT_ADMITTED`: publisher
episode identity is available, but scene/layout identity is not. Normalized
content overlap across the selected splits has not been evaluated. Do not treat
episode-held validation as scene-independent or product evidence.

No model features were extracted, no readout was fit, no reserved media was
downloaded, and no GPU was used. The previous cabinet-left artifacts and the
metadata-only screen remain unchanged.

## Frozen source and split

- Dataset: NVIDIA `nvidia/PhysicalAI-Robotics-Manipulation-SingleArm`, subset
  `panda-stack-platforms`, revision
  `73c00c18e7ca108e91846294b20a1d7cd5de9340`.
- Publisher declares CC BY 4.0; attribution is required. No media is included
  in Git.
- Source commit: `b744d1c407c6c4c00208b67f371e421c14a890a7`. The acquisition
  script was an uncommitted working-tree file; its pre-run hash was not
  captured. After the run, duplicate-ID validation and a formatting-only Ruff
  fix were applied; the current script SHA-256 is
  `dacc08e20d83ecee2ad8b389a4cf92fdf2901f43e450267f6f727ed580d5504c`.
- Split manifest:
  `manifests/pretrained_reuse/physicalai-singlearm-stack-platforms-split-v3-20261102.json`,
  SHA-256 `3a2849afaa3cefa414c9dbfc094976e0f1cc48ad0e9636d65573fa306d1ed967`.
- Deterministic split seed is 17, grouped by publisher `episode_index`: 12,340
  train, 2,645 validation, and 2,644 reserved groups. The selected pilot is
  the first 700 train and 250 validation IDs by the manifest's frozen rank.
- Pilot train and validation ID-order hashes are respectively
  `de1ccf39133dec49ad047069c5002a19a8a418d1e35467cd46178071464bfa17` and
  `fe40953ebf5961b9158eed521bf265107c3bb596283a0621ec0ed993dd39eafc`.
- The full inventory records zero exact cross-split video-SHA and Parquet-SHA
  overlap. Its upstream metadata has no scene/layout/environment key;
  scene-level independence is UNKNOWN.

## Acquisition and integrity

The exact command was:

```powershell
python scripts/download_physicalai_stack_platforms_pilot.py `
  --manifest manifests/pretrained_reuse/physicalai-singlearm-stack-platforms-split-v3-20261102.json `
  --output-dir C:\CodexArtifacts\pretrained-reuse-r2\physicalai-stack-platforms-pilot-20261102 `
  --workers 8
```

The run started at 2026-11-02 14:06:52 JST and completed at 14:16:04 JST.
The receipt reports 550.391 seconds, 1,900 files (950 videos and 950
Parquets), and 192,700,909 bytes. The receipt SHA-256 is
`1a2b09ad683fe5da38fec74ac937aefaa9c5a1b2ed969285478a0f724f13a498`. All
1,900 local files were independently re-read and matched the receipt's byte
sizes and SHA-256 values; zero files were missing or mismatched. The receipt
states `reserved_holdout_payload_downloaded: false`.

Three early transfer intervals measured approximately 0.17, 0.28, and 0.34
MB/s (mean 0.26 MB/s). The initial remaining-time estimate was about 12
minutes; later 60-second samples showed faster sustained transfer, and the
actual run completed in about 9 minutes 10 seconds. This rate is specific to
this local route and run.

At the final resource check, C: had about 7.42 GiB free. The RTX 3080 Laptop
GPU remained allocated to the separate Orbitune process (15,565/16,384 MiB); it
was not stopped or modified.

## Verification and next gate

The split-builder and downloader CPU tests pass: 7 passed. The full repository
suite passes: 2,757 passed, 11 skipped. Ruff and all CI manifest/catalog
validation commands pass. The downloader fails closed on size/hash mismatch,
refuses duplicated or overlapping pilot IDs before network access, and reuses
only already verified files. The receipt and media remain outside Git at the
artifact path above.

Before any feature extraction or fitting, freeze the action-prototype task
protocol and verify normalized-content overlap on the selected train/validation
media. Even if those checks pass, report only episode-held, same-source
development results unless scene/layout independence can be established. Keep
the source unadmitted and do not access the reserved partition.

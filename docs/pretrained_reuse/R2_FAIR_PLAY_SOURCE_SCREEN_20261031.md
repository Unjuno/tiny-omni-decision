# FAIR-Play audiovisual correspondence source screen (2026-10-31)

## Disposition

FAIR-Play is **not admitted for fitting or quality evaluation** in this R2
workstream at this time. Its clip-pair correspondence task is technically
runnable with already audited frozen audio/video encoders, but its released
split metadata does not currently meet the project's independent-group and
validation-coverage gates. The existing Perception Test work already evaluates
the same four-way audio-to-video recording correspondence task, so FAIR-Play
would not supply a new task family by itself.

## Pinned source and split evidence

- FAIR-Play repository: `facebookresearch/FAIR-Play` at
  `2666d6caf2d126ae43a84841e94e0c9ae24b3e2d`.
- PseudoBinaural split repository:
  `SheldonTsui/PseudoBinaural_CVPR2021` at
  `87f7be08d41748995c561ad1dc7bda453d4ee553`.
- Both public repository metadata records state CC BY 4.0. This screen does not
  establish rights to any underlying participant or media beyond those stated
  records; preserve attribution and review the actual redistribution terms
  before broader use.
- Each of the five published PseudoBinaural folds has 1,497 train, 187
  validation, and 187 test clip IDs; within-fold train/validation ID overlap is
  zero. The published split description claims scene separation, but the
  pinned split files do not include scene keys or a visible generation script.
  The independent scene-group count is therefore UNKNOWN; even its maximum
  possible validation count, 187, falls below the provisional 200-group
  minimum.
- Different folds are not independent replicates. The recorded five-fold
  metadata audit finds substantial train/validation asset overlap across folds
  and pairwise validation-ID overlap. Do not pool folds as if they were
  independent or use another fold to tune the same asset.
- The fixed `split1` train/validation/test ID lists have SHA-256 values
  `B47A45AF24CAB9657D4096A9F960C2E5C8AFC38AF15F93862EE871F4F98B3997`,
  `D615FFEA53D8C2F1B257E980D3C292AF13C163F9830031164E052E8590EB1910`, and
  `AFE2704C2B91E2F355078333ECE12BF684D435ECE7990086555BD048B21B4581`.
  Metadata audit reports are retained outside Git in the `fair-play-split-audit`
  attempt directories.

## Three-pair acquisition and runtime sample

Three predeclared `split1/train` audio/video pairs were fetched to a separate
artifact directory; no validation or test media was opened. Their combined
size was 173,249,235 bytes. Transfer rates were 16.867, 14.661 and 14.257 MiB/s,
mean 15.262 MiB/s. Extrapolating the sample to 443 pairs (256 train + 187
validation) gives about 24.4 GiB and 26.6 minutes; this is a rough projection,
not an observed bulk-transfer time. At the measured C: free space of about
28.6 GiB, the projected download would leave little operational reserve, while
the group gate would still fail. The larger official train+validation set
would exceed local free space.

For the three train pairs, the pinned VideoPrism-B produced finite framewise
features with the existing CPU pipeline and the pinned EfficientAT produced
finite audio embeddings on the local RTX 3080 under the existing GPU guard.
These three pairs verify only media decode and feature plumbing. No labels were
used, no scorer was fitted, and no task metric was produced. The six raw media
files, timing receipt, extraction scripts, feature files, and hashes are
retained at
`C:\CodexArtifacts\pretrained-reuse-r2\fair-play-runtime-samples-20261031`.

## Next admissible step

Do not bulk-download or fit on the current folds. Reconsider only if an
independently verifiable scene/recording-group split with at least the
provisional validation-group floor and a predeclared task not duplicating the
existing correspondence evaluation becomes available. Keep this source screen
separate from the Perception Test correspondence artifacts.

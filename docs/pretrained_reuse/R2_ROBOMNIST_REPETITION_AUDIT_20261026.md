# RoboMNIST repetition and payload feasibility audit

**Status: source/runtime feasibility evidence only; no model fitting or quality result.**

## Scope and provenance

This continuation follows the R2 admission gate in the current roadmap. It
rechecked the pinned Figshare article/version 1 metadata for DOI
`10.6084/m9.figshare.28179383.v1`, the author's pinned repository commit
`70f5e6393acf6128a7ceb7afacf4463060626ca3`, and its CC BY 4.0 release terms.
The original source screen remains at
[`R2_ROBOMNIST_SOURCE_20261014.md`](R2_ROBOMNIST_SOURCE_20261014.md).

No sealed audit, training corpus, model checkpoint, or existing run was read or
changed. No full archive was downloaded. The 60 ZIP central directories were
read with bounded HTTPS Range requests: 8,621,356 bytes transferred from
56,403,232,879 bytes of declared archives. HTTP 206, exact range/length, TLS,
and transfer caps were enforced. The authoritative corrected directory receipt
is outside Git at
`C:/CodexArtifacts/pretrained-reuse-r2/robomnist-directory-audit-20261026-attempt-04.json`;
SHA-256: `538D526BFA049D4CEF29568BCBE06B32A96A303A2C229D10CF0420641932D0CB`.

An earlier attempt-02 receipt contained a bad local-header-offset field decode;
it was not used to access payload. The parser was corrected to read the ZIP
central-directory local-header offset at its specified field, regression-tested
for distinct offsets, and all 60 directories were re-read for attempt-04. Each
selected payload member was then independently checked against its local header,
declared size, and central-directory CRC-32. Whole-archive MD5 remains
unverified; `31_Ar2_Vmed_Ac1.zip` has no computed MD5 in the pinned metadata.

## What the full directory inventory establishes

The 60 archives contain 2,135 complete **filename-derived repetition groups**.
Every group has the expected seven runtime inputs (three camera videos, three
microphone recordings, and one CSI JSON) plus one position JSON treated only as
supervision. Counts per archive range from 32 to 44, even though the upstream
README describes 32 repetitions per class:

| Filename-derived groups per archive | Archive count |
| ---: | ---: |
| 32 | 1 |
| 34 | 12 |
| 35 | 28 |
| 36 | 6 |
| 37 | 8 |
| 38 | 1 |
| 39 | 1 |
| 40 | 1 |
| 42 | 1 |
| 44 | 1 |

This discrepancy is unresolved. The group identifiers do not establish
independent participants or sessions; filenames encode robot, velocity, action,
repetition, and an uncertainty code, but upstream session provenance was not
found. Consequently, “2,135 groups” is not a claim of 2,135 independent
evaluation units.

The exact inventory records member names, compressed/uncompressed sizes, CRC,
and local header offsets. It is not a train/validation/test split and does not
claim content deduplication across groups.

## One fixed repetition: payload, decoding, and timestamps

Before reading payload, the selection rule was fixed as the lexicographically
smallest complete group ID in the first archive. This selected
`Ar1_Vhigh_Ac0_Rp01_UNC062`. Only its seven inputs and supervision-only
position file were range-fetched; 26,129,831 compressed bytes were transferred
and 37,411,951 bytes were decompressed in memory. Each member passed a CRC-32
check. No payload file was retained.

PyAV 18.1.0 decoded all three video/audio views:

| Stream | Result |
| --- | --- |
| Each camera | MPEG-4, 2560×720, 450 decoded frames, 0–14.9667 s PTS, 15.0 s duration |
| Each microphone | Mono PCM S16LE, 44.1 kHz; durations 14.9304 s, 14.9537 s, 14.9769 s |
| Each of 3 CSI time arrays | 450 points, strictly increasing, 14.966323 s span, median interval 33.329434 ms |
| Each of 2 position-supervision time arrays | Same 450-point span and median interval; strictly increasing |

This single group supports that the released formats are decodable and the
video/CSI/position sampling is plausibly aligned near 30 Hz. It does not prove
the exact sensor-to-video clock offset, audio start-time alignment, or
cross-repetition independence. The audio comparison here is duration-only.
No labels, positions, or source filenames are model inputs; the position file
was used only to inspect schema and timestamp structure.

The outside-Git payload receipt is
`C:/CodexArtifacts/pretrained-reuse-r2/robomnist-payload-audit-20261026-attempt-04.json`;
SHA-256: `BBBFA35795CF086CBEAE6B1A95A61D1E588440F5FE149A62ED0C43C96F4BC8C6`.
It records the exact per-member decode and timestamp summaries.

## Candidate task manifests and remaining gate

RoboMNIST is now a technically viable small source for a controlled
audio/video correspondence task, but **it is not yet admitted for model fitting
or quality evaluation**. A metadata-only manifest generator creates the
candidate question “Do these audio and video clips show the same action
category?” with two supplied options. It pairs different repetition groups
within each fixed robot/velocity block: a positive has the same action and a
negative has a different action. Each selected video asset and each selected
audio asset appears once with each target in its split, so either modality
alone cannot identify the pair label from asset identity or class frequency.

The deterministic seed-17 split holds out `floor(groups/8)` groups for each of
validation and sealed audit within each of the 60 source archive classes; the
remaining groups are training candidates. For exact per-block label balancing,
105 groups beyond the smallest action support were left unused. The resulting
candidate manifests contain 3,100 train pairs (1,550 per target), 480
validation pairs (240 per target), and 480 sealed-audit pairs (240 per target).
The manifests reference 1,550 / 240 / 240 distinct underlying repetition
groups, respectively, with zero group overlap. The sealed-audit manifest is
stored in a separate sibling directory and was not loaded or evaluated.

These are split- and task-construction checks, not model controls or quality
results. A future runtime must prove that the loader sends only the fixed
question, options, and decoded audio/video tensors; the archive/member names,
group IDs, robot/velocity/action metadata, position supervision, and target
must not enter model inputs. Question-only, audio-only, video-only, and
pair-shuffle controls still need measured evaluation. Group identifiers are
filename-derived repetitions and do not establish independent collection
sessions, so the current validation and audit candidates cannot support a
session-generalization claim. The manifest schema separates the model-visible
`model_input` (`question` and `options`) from the target and `source_refs`; the
runtime adapter must preserve that boundary when decoding the referenced files.

The outside-Git manifest outputs are under
`C:/CodexArtifacts/pretrained-reuse-r2/robomnist-av-correspondence-seed17-attempt-03/`
and the separate sealed path
`C:/CodexArtifacts/pretrained-reuse-r2/robomnist-av-correspondence-sealed-seed17-attempt-03/`.
Train SHA-256: `38D374B37CB80031E35045084F286F8272D66146E352EA51BD2E7B6046C7F1EE`;
validation SHA-256: `40A0FCA9638080BE890DFD4673959F01381D5DF7B8ECA4DBAAF8306EE10B0E12`;
sealed-audit SHA-256: `F520A6539D173F486B7CC25A55498B4C764941F40F66A4758D6689C8051BCC93`.
Manifest summary SHA-256: `6E90160CBCC630051B7C3E5A62599788990DA238FBC6BF4438497D3AC8460330`.

Before fitting, the next gate is to establish repeat/session identity and a
group-disjoint split, freeze matched/mismatched task construction, and confirm
that unimodal shortcut controls are balanced by construction. Any missing
whole-archive digest and same-lab scope must remain explicit. A one-repetition
decode is not a quality result and cannot support the product's 95–98% claim.

The full objective remains unchanged: a small, pre-quantization multimodal
Decision model with task-specific quality evidence. This report closes neither
R2 M-2 nor the broader R2 exit, and it does not justify a shared Lens or merge.

## Reproduction

```powershell
python -m scripts.audit_robomnist_archive_directories `
  --output C:/CodexArtifacts/pretrained-reuse-r2/robomnist-directory-audit-20261026-attempt-04.json

python -m scripts.audit_robomnist_repetition_payload `
  --directory-audit C:/CodexArtifacts/pretrained-reuse-r2/robomnist-directory-audit-20261026-attempt-04.json `
  --version-metadata C:/CodexArtifacts/pretrained-reuse-r2/robomnist-source-audit-01/article-version-1.json `
  --output C:/CodexArtifacts/pretrained-reuse-r2/robomnist-payload-audit-20261026-attempt-04.json

python -m scripts.build_robomnist_av_correspondence_manifests `
  --directory-audit C:/CodexArtifacts/pretrained-reuse-r2/robomnist-directory-audit-20261026-attempt-04.json `
  --output-dir C:/CodexArtifacts/pretrained-reuse-r2/robomnist-av-correspondence-seed17-attempt-03 `
  --sealed-output-dir C:/CodexArtifacts/pretrained-reuse-r2/robomnist-av-correspondence-sealed-seed17-attempt-03
```

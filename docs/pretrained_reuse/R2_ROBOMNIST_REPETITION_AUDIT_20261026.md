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

## Decision-task implication and remaining gate

RoboMNIST is now a technically viable small source for a controlled
audio/video/CSI correspondence experiment, but **it is not yet admitted for
training or quality evaluation**. A candidate task for a future preregistration
is whether an audio recording and a video depict the same action category.
Matched and mismatched pairs could be formed from disjoint repetitions, with
each input asset balanced across match labels. This would let us check that
audio-only, video-only, and question-only controls are at chance while the
joint input is informative. The proposal has not been implemented or evaluated.

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
```

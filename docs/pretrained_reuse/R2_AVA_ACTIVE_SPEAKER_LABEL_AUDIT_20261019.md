# AVA ActiveSpeaker TRAIN label audit — 2026-10-19

## Disposition

AVA ActiveSpeaker remains **REVIEW / NOT_ADMITTED**. The official TRAIN label
archive is pinned and audited without extracting files. No audio/video media,
model weights, or challenge-test archive were fetched. No feature extraction,
model fit, or quality evaluation was run.

The archive linked by the official page as validation contains internal member
paths named `ava_activespeaker_test_v1.0`. During format discovery, the first
CSV member (22,807 rows) was parsed before that path was noticed. The archive is
therefore quarantined: **do not use it for validation, model selection, or any
metric**. No other member payload was parsed. No challenge-test archive was
fetched. This exposure is recorded rather than silently treated as a clean
validation set.

## TRAIN-only evidence

The official AVA page lists CC BY 4.0 for AVA datasets and links separate
ActiveSpeaker TRAIN/validation archives. The paper/source describes an
audio-visual speaker-to-face association task, but the original assets are
YouTube videos and their individual media rights remain unresolved.

| Item | Observed value |
| --- | ---: |
| TRAIN archive | `ava_activespeaker_train_v1.0.tar.bz2` |
| Compressed bytes | 18,187,972 |
| SHA-256 | `528594517178b2d04ab28914d35c95c3e2bdca5c46e7af359921968bef60c3d5` |
| TRAIN video members | 120 |
| Annotation rows | 2,676,314 |
| `NOT_SPEAKING` | 1,969,134 |
| `SPEAKING_AUDIBLE` | 682,404 |
| `SPEAKING_NOT_AUDIBLE` | 24,776 |
| Invalid rows under audited structural checks | 0 |

The row schema is video ID, fractional-second timestamp, normalized face box,
label, and within-video entity track ID. Every TRAIN member ID matched its rows;
timestamps and boxes were finite and in bounds. The audit read only the pinned
TRAIN archive and retains per-video timestamp/track counts in the external JSON
receipt at
`C:/CodexArtifacts/pretrained-reuse-r2/ava-activespeaker-label-screen-01/train-label-audit.json`.

## Source discrepancy and limitations

The official web page documents labels as `SPEAKING_AND_AUDIBLE`,
`SPEAKING_BUT_NOT_AUDIBLE`, and `NOT_SPEAKING`. The TRAIN archive actually
contains `SPEAKING_AUDIBLE`, `SPEAKING_NOT_AUDIBLE`, and `NOT_SPEAKING`.
The auditor preserves the observed strings and does not map them to the
documented names. Reconcile this schema with the dataset owner/source release
before building labels or a Decision adapter.

The archive linked as validation is 5,097,948 bytes (SHA-256
`ced431ed2006deacdacd8954d309c5b780179866b0f63d5d17d4df8a8ec81622`). Its
member paths are test-named; because one label row was observed, no rows or IDs
from it are used in this project. Consequently, train/validation ID disjointness,
source grouping, duplicate media and a clean selection set are still
unverified. The public page claims the labels share AVA Actions' train/validation
partition, but that statement does not repair the local archive ambiguity.

The proposed task would take synchronized audio/video and a supplied face-track
proposal, then predict speaking-and-audible status. This may provide a useful
cross-modal association task, but face boxes are oracle inputs and the labels do
not prove that both streams are always necessary. Any future experiment needs
media rights clearance, an independently verified clean validation source,
video-grouped splits, and matched audio-only/video-only/joint controls. Do not
use the contaminated validation-link archive or the challenge-test labels.

## Reproduction

```powershell
py -3.11 -m scripts.audit_ava_active_speaker_train_labels `
  --train-archive C:\CodexArtifacts\pretrained-reuse-r2\ava-activespeaker-label-screen-01\ava_activespeaker_train_v1.0.tar.bz2 `
  --output C:\CodexArtifacts\pretrained-reuse-r2\ava-activespeaker-label-screen-01\NEW_OUTPUT.json
```

The CLI deliberately refuses to overwrite an existing receipt; replace
`NEW_OUTPUT.json` with a fresh path for a replay.

The script accepts only the exactly named TRAIN archive, checks every member
path, streams CSV rows without extraction, bounds invalid-row details, and has
regression tests showing validation-named archives are rejected before open.

## Sources

- [Official AVA download page and archive links](https://research.google.com/ava/download/).
- [AVA ActiveSpeaker paper summary](https://research.google/pubs/ava-activespeaker-an-audio-visual-dataset-for-active-speaker-detection/).
- [Official AVA dataset overview](https://research.google.com/ava/).

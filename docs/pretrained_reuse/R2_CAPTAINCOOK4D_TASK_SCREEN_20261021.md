# CaptainCook4D task and split screen — 2026-10-21

## Decision

CaptainCook4D is a **provisional text+video task candidate** for checking
whether an observed procedural step matches its written step description. It
is **not admitted as an audio+video complementarity task**: its step-error
labels do not state that audio is needed, and audio necessity cannot be
validated from the label schema alone.

No media was downloaded or decoded, no model was run, no training occurred,
and no official test metrics or project sealed-audit results were used. The
official annotations and downloader metadata were inspected at pinned source
revisions; the existing model and all previous run artifacts remain unchanged.

## Source and rights

The official project describes 384 recordings (94.5 hours), 5.3K step
annotations, 10K fine-grained action annotations, and supervised error
recognition, multi-step localization, and procedure-learning tasks. The project
states that its data are released under Apache-2.0 and that participants gave
written consent. This source is suitable for a bounded research screen; it
does not establish that all modalities are required.

Pinned source revisions:

| Component | Revision |
|---|---|
| Official annotations | a8a920a3293c4db27099a20ddbe3a3a9be1283e3 |
| Official downloader and stream metadata | cf248fe2139fc6e37f9737141ec719948fbd1747 |
| Official project repository | 1137b884e038cc128589c18eab215024018fb642 |

The external CPU audit receipt is
C:\CodexArtifacts\pretrained-reuse-r2\captaincook4d-source-screen-attempt-01\source-audit.json,
SHA-256 FC3A679040DF6D547D2460CC6D623A16703DECDFA98B55709DF423E5DF76B66B.
The annotation and downloader checkouts are under
C:\CodexArtifacts\pretrained-reuse-r2\captaincook4d-annotations-screen-a8a920a
and C:\CodexArtifacts\pretrained-reuse-r2\captaincook4d-downloader-screen-cf248fe.
Only annotations, split metadata, video metadata, and download-link metadata
were fetched.

## Task and split audit

The step table provides a written action/step description and a has_errors
label. A meaningful Decision form is: *Does the visible action match the
provided step?* with probabilities over consistent / inconsistent.
The candidate inputs are step text and video segment. Audio is excluded
pending evidence that it adds target-relevant information. Text-only and
video-only controls would be required alongside the joint condition.

The official person split has 204 train recordings and 84 validation
recordings, with no recording-ID overlap. The audit found one person shared
across splits: training recording 15_33 and validation recordings associated
with participant ID 8. Removing that one training recording gives 203
training recordings and zero person-ID overlap. Train/validation still share
5 environments and 23 of 24 activities, so this is a same-recipe,
mostly-same-environment person holdout, not an unseen-recipe or unseen-site
test. The validation side has only two participant IDs, one represented by
three recordings; group-level uncertainty will therefore be weak.

For the unmodified official train/validation selection, there are 2,975 train
step rows (1,060 labeled errors) and 1,247 validation step rows (382 labeled
errors). The official downloader metadata lists GoPro 360p video for 203/204
train and 84/84 validation recordings. HoloLens synchronized PV video is listed
for 185/204 train and 57/84 validation recordings. These are link-availability
counts; they do not verify local media contents, audio streams, timing, or
annotation-to-frame alignment.

On 2026-10-21, read-only HTTP HEAD checks of the listed GoPro 360p links for
training recording `10_16` (person 5) and validation recording `10_31`
(person 8) both returned HTTP 404 from Box. No media bytes were downloaded.
Therefore the metadata availability counts above do not establish that the
media is currently reachable. The media-alignment smoke is blocked until the
project provides working official links or an authorized mirror.

The original split and labels were used only for this source screen. The
combined split JSON was parsed as a whole, but the audit code only reads its
`train` and `val` keys; it does not inspect the `test` key. The shared step
annotation CSV was streamed, but rows outside the official train/validation
ID allowlist were discarded before labels were counted or results retained.
No test metrics were computed, and test IDs/labels are not in the receipt or
any model evaluation. This was a source-metadata screen, so it does not meet a
strict “test files were never opened” standard. No media-file identity or
normalized-content hashes have been checked.

## Recommended next gate

Before model evaluation, create a new immutable task manifest that:

1. removes training recording 15_33 or uses a separately defined
   participant-disjoint split;
2. binds each row to one exact video segment, written step, target, and source
   recording;
3. confirms video/timestamp alignment on a small media smoke through the
   existing local compute guard;
4. checks exact and normalized content overlap across included splits;
5. fixes the selection and scoring rules before any model comparison.

This small split can support plumbing and exploratory development. Its two
validation participants and broad recipe overlap are insufficient for a
strong high-accuracy or out-of-recipe claim. Do not compute or inspect test
metrics. Do not spend GPU time until the task manifest and a useful
group-level validation design are fixed.

## Reproduction and verification

    python -m scripts.audit_captaincook4d_source --annotations C:\CodexArtifacts\pretrained-reuse-r2\captaincook4d-annotations-screen-a8a920a --downloader-metadata C:\CodexArtifacts\pretrained-reuse-r2\captaincook4d-downloader-screen-cf248fe\metadata\download_links.json --output C:\CodexArtifacts\pretrained-reuse-r2\captaincook4d-source-screen-attempt-01\source-audit.json

CPU regression checks: 2 passed; Ruff passed for the new audit module and
tests. No GPU, training, model checkpoint, official test metric, or sealed
audit was accessed.

## Official references

- [CaptainCook4D project page: scope, dataset license, consent](https://captaincook4d.github.io/captain-cook/)
- [Official annotations and split repository](https://github.com/CaptainCook4D/annotations)
- [Official downloader and available streams](https://github.com/CaptainCook4D/downloader)
- [Official multimodal error-recognition baseline](https://github.com/CaptainCook4D/error_recognition)

# Ego4D Social license-draft review

## Official task fit

The official Social benchmark defines “Talking to me” as deciding, for a
visible tracked face in a video/audio segment, whether that person is speaking
to the camera wearer. Its labels combine active-speaker annotations with a
directed-to-wearer judgment. This is a substantially better R2 Decision-task
candidate than generic sound-event presence because the decision concerns a
time-localized person and social direction.

Official task description: <https://ego4d-data.org/docs/benchmarks/social/>.

## Public draft terms checked

The official Ego4D site links a 36-page draft license:
<https://ego4d-data.org/pdfs/Ego4D-Licenses-Draft.pdf>.
The draft contains separate agreement templates for multiple licensors and
jurisdictions. In the reviewed templates, the grant is non-exclusive and
non-transferable, tied to a defined purpose, and begins only after execution by
the authorized individual or organization. The draft describes research,
training, evaluation, and improvement of machine-learning models for detecting,
classifying, recognizing, retrieving, or understanding objects/events/places/
activities. It says qualifying derived software and models may be used for
academic, commercial, or noncommercial purposes, while the underlying database
must not be redistributed or embedded in a product. Attribution and data
security duties apply.

This narrows the earlier “terms unavailable” finding: the public draft suggests
commercial use of qualifying derived models is possible **after an applicable
agreement is executed**. It does not grant permission to access annotations or
media before signing, and it does not establish which licensor agreement covers
the exact TTM subset or whether the intended licensee qualifies to sign.

## Gate status

- Task: promising, not admitted.
- Public draft: reviewed, not an executed license.
- Required before access: identify the applicable licensee and agreement(s),
  have the authorized person review and execute them, then receive valid access
  credentials.
- Still unknown: exact TTM annotation revision/hash, usable group counts by
  split, locally available media, media coverage and synchronization, and
  participant/video/source overlap for a project-specific train/validation
  split.
- No annotations or media were downloaded, opened, or hashed in this review.
  No training or evaluation occurred.

If access is authorized later, first perform a metadata-only split/coverage
preflight on an approved benchmark subset, grouped by the original source video
and camera wearer. Do not begin fitting until the task groups, media identities,
audio/video synchronization, and rights are verified.

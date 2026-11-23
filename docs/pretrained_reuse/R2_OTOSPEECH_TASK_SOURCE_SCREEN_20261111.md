# otoSpeech Task source/task screen — 2026-11-11

## Disposition

Keep `otoSpeech-full-duplex-task-oriented-20h` at `REVIEW_NOT_ADMITTED` for
Path A fitting or quality evaluation. It is a promising task-oriented
audio/visual interaction source, but the released cohort is too small and
access is gated. No access request was submitted, no gated files or session
content were opened, and no audio, image, label, model, validation, or sealed
audit payload was downloaded or used.

## Pinned public metadata

- Dataset: `otoearth/otoSpeech-full-duplex-task-oriented-20h`.
- Hugging Face dataset API revision: `383254132dbd8e9465da91f2fe683728c706923f`.
- Last modified: `2026-08-31T13:02:19Z`.
- Public metadata reports `cc-by-4.0`, `gated: auto`, and the dataset page
  requires agreeing to share Hugging Face account information to request
  access. The page states that access is manually reviewed. No terms were
  accepted and no request was sent.
- The publisher describes 20.005954 hours, 58 sessions, seven collaborative
  task types, 11,025 timestamped events (6,115 primary interaction events),
  two-speaker channel-separated 48 kHz audio, synchronized task-state events,
  task stimuli, and reference answers for some tasks.
- The page reports a 3.88 GB full release. This is metadata only; no release
  shard or file tree was accessed.

## R2 gate assessment

The source is conceptually relevant: task state records what each speaker saw
and did, and some tasks intentionally give the speakers different views. The
publisher's `Spot the Difference` task has a checkable list of visual
differences and describes audio as the bridge between those views. `Tangram
Direction` includes target arrangements and asymmetric access. These are
plausible seeds for both-input-needed Decision tasks, but the dataset does not
publish a model-ready supplied-option Decision protocol for them; a conversion
would need to be designed and audited before any fit.

The required independent-group floor cannot be met by this release as
described. There are only 58 sessions in total, fewer than the roadmap's 200
independent source-group minimum even if every session were independent. The
task counts are more limiting: Spot the Difference has 18 sessions over only
10 scene pairs, and Tangram Direction has 3 sessions over 10 scenarios. Group
identity overlap across sessions is not established by the public summary, so
58 is an upper bound, not a verified independent-group count. The minimum
1,000-example/200-group R2 gate therefore fails before any data acquisition.

The CC BY 4.0 declaration applies to released repository contents according to
the card. Photo Talk additionally indexes 82 third-party Flickr photographs
under mixed CC BY, CC BY-SA, and CC BY-ND licenses. Those stimuli would have to
be excluded or individually audited even if access were approved. A future
decision to request gated access would also require the user's own authorization
to disclose account/work information; this screen does not imply that consent.

## Decision and next step

Do not request access, download the 3.88 GB release, or fit a readout from this
source under the current R2 thresholds. The result is a candidate rejection on
group scale and unresolved task protocol, not a negative model-quality result.
Continue searching for an openly accessible or explicitly approved,
rights-compatible synchronized task with at least 1,000 examples and 200
independent original groups, a fixed option-decision target, and predeclared
unimodal/missing/shuffle/time-offset controls. Keep the existing Path A runtime
as an engineering component; do not infer quality from it.

## Sources and method

- Publisher-authored dataset article, published 2026-08-31:
  <https://huggingface.co/blog/otoearth/otospeech-task>
- Publisher's Hugging Face dataset card and public metadata:
  <https://huggingface.co/datasets/otoearth/otoSpeech-full-duplex-task-oriented-20h>
- API metadata endpoint queried read-only on 2026-11-11:
  <https://huggingface.co/api/datasets/otoearth/otoSpeech-full-duplex-task-oriented-20h>

Only the public card/article and dataset API metadata were examined. The API
identified the revision above. The gated repository's file list, task shards,
audio, stimuli, event annotations, and labels were not accessed.

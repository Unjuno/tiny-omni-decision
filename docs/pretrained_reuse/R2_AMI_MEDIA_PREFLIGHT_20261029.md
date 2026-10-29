# AMI media and annotation preflight — 2026-10-29

## Outcome

AMI remains `REVIEW_NOT_ADMITTED` for R2-M. The official mirror has a small,
technically decodable synchronized audio/video subset, but this does not yet
prove that the proposed label is valid, that both modalities are necessary, or
that the available independent meeting groups support a meaningful comparison.
Do not fit a model or claim quality from this preflight.

## Reconciliation with existing AMI work

This is a small alternate low-bitrate/mixed-audio check, not a replacement for
the more complete work already in [Draft PR #19](https://github.com/Unjuno/tiny-omni-decision/pull/19)
at head `406f003a128a7991ec39dcb15608b381f003add4`. Its report and receipts
cover 4 selected SA/train and 5 SB/validation parent sets, 214 train / 111
validation candidate windows, exact media hashes, selected annotation files,
global SA/SB participant-ID disjointness, and 288 three-point audio/video PTS
observations. The train-only visual screen found at least 2 of 12 sampled
positive target-camera labels showing an empty room. Its later speaker-channel
and timing audits still leave visible-speaker truth and sufficient independent
validation coverage unresolved. The follow-on branches
`codex/r2-ami-speaker-channel-preflight-20261022` (`cb020322a229ea31a7d2773717ba6487a622a376`),
`codex/r2-ami-rm-media-preflight-20261023` (`9966302c06ebacae30d292b88d13c98d03a5d021`),
and `codex/r2-ami-multipt-timing-gate-20261023` (`e25d21300850126d408ed837ef7437c99d8f3829`)
were also read before this update. Reuse that evidence; do not repeat its
speaker-channel acquisition, timing sample, or six-meeting RM decode.

The external artifact also contains a broader 213-row SA training visual-
presence review: 168 `face_visibly_framed`, 21 `limited_or_ambiguous`, and 24
`no_clear_person`. It was one qualitative reviewer over three still frames per
row (15/50/85%); it does not verify speaker identity, speech alignment,
synchronization, or label correctness and is not a prevalence estimate. The
review JSON SHA-256 is
`E344E81BC6B60DA0D65C5A96387225C47C5A0F7AFD6F2BD5DC43E4EF096F971F`; its input
manifest SHA-256 is
`5A1BF48D478C889002C2C238E0CED7F376F2494EF490848A54A01C8FA2B61882`. Keep
those rows as training-only source diagnostics; do not filter evaluation rows
or claim the screen makes the task labels valid.

The new smoke is limited to one train meeting using the mixed Room-Lapel track
and official low-size RM streams. It adds a direct-stream decoder check and
exact `Content-Length`/ETag evidence for one submeeting in each scenario
partition. It does not address the earlier visibility failures or provide a
new quality comparison. The existing SA/SB participant-channel cohort is the
stronger task candidate if its remaining label gates can be resolved.

A follow-up listing check found eight individual low-bitrate RM audio tracks
for ES2002a (`Headset-0..3` and `Lapel-0..3`); the official directory also
lists their WAV counterparts. `ES2002a.Lapel-0.rm` returned HTTP 200 with
2,660,658 bytes and ETag `"289932-3f389de7dfdc0"`. A direct CPU stream smoke
decoded its first SIPR frame (16 kHz mono, 160 samples; in-memory frame SHA-256
`adeecefccf7184a502e0addee766f1d2a8f140cac12591bb1e6e916099ad450d`). The
official [signal description](https://groups.inf.ed.ac.uk/ami/corpus/signals.shtml)
states that the individual audio files are per-channel and its ES2002a table
maps channel 0 to MEE006 / Closeup1. The official
[data-problems table](https://groups.inf.ed.ac.uk/ami/corpus/dataproblems.shtml)
notes that this participant's headset was worn incorrectly but its lapel
recording is okay. Thus `Lapel-0` is the appropriate channel-matched source
candidate; the speaker-turn annotation and temporal AV alignment still need
validation. No full track or feature cache was saved. This improves source
availability evidence only and does not change `REVIEW_NOT_ADMITTED` or close
the task/label/validation gates.

## Source and exact media inventory

The source remains the official AMI mirror and its scenario-only split. The
three meetings below were checked with directory listings and HTTP `HEAD`
requests; complete media files were not saved. `Content-Length` is the exact
byte count reported by the mirror. HTTP ETags are recorded in the companion
[manifest](AMI_ES_MEDIA_PREFLIGHT_20261029.json); they are validators, not
cryptographic hashes.

| Split role | Meeting | Room audio | Four close-up videos | Total bytes |
| --- | --- | ---: | ---: | ---: |
| Train example | ES2002a | 2,660,658 | 32,246,173 | 34,906,831 |
| Validation example | ES2011a | 2,329,668 | 28,909,064 | 31,238,732 |
| Unexposed evaluation candidate | ES2004a | 2,195,266 | 29,991,292 | 32,186,558 |
| **Total for these three submeetings only** |  |  |  | **98,332,121** |

This is one submeeting from each of three scenario groups, not a complete
training/validation corpus. The official scenario-only split has 11 train, 2
validation, and 2 evaluation groups. Even if many turns are extracted, the
small number of independent held-out meeting groups limits uncertainty. The
sample is not sufficient for the roadmap's provisional 200-independent-scene
floor or a robust group-level quality claim.

ES2003a was checked as an alternative validation meeting. Its official video
listing has Closeup2–4, and an HTTP `HEAD` for `ES2003a.Closeup1.rm` returned
404. It was not used. ES2011a has all four close-ups listed and the audio plus
all four video files returned successful `HEAD` responses.

## Actual decoder smoke

Using Python 3.11 and PyAV 18.1.0, one frame was decoded from the ES2002a train
Room-Lapel audio, one from `Lapel-0`, and one from its Closeup1 video by reading
the official HTTP URLs directly. No media files or feature caches were written,
and this was a CPU-only process.

- Video: RV40, 360 × 284, 15 fps; decoded RGB frame SHA-256
  `a3be406da1cee3ffd175f1a20c2c60f9057cc33e4bf16fe144dd71952d86159f`.
- Audio: SIPR, 16 kHz, mono; decoded first-frame SHA-256
  `5f48b7357acaeef9046a423c6bdebbf3ea62e643a39941bdcebc49626a7e8210`.
- Individual-track availability probe: `Lapel-0` is also SIPR, 16 kHz, mono;
  first frame has 160 samples and SHA-256
  `adeecefccf7184a502e0addee766f1d2a8f140cac12591bb1e6e916099ad450d`.
  Official per-channel naming/mapping makes it channel 0 (MEE006 / Closeup1);
  its turn labels and timestamp alignment remain unverified.
- The decoded-frame hashes identify those in-memory frames only; they are not
  hashes of the source files.
- For ES2002a Closeup1, the video stream starts at 0 and reports 1,230.321 s.
  The Room-Lapel audio starts at 0 and reports 1,272.960 s. The other three
  close-up streams report 1,248.161–1,248.641 s. Any aligned task must restrict
  timestamps to the common valid interval and verify alignment against the
  official annotation timing; equal zero start timestamps alone do not prove
  sample-level synchronization.

The official low-size video is therefore technically decodable, but at
360 × 284 and 15 fps its usefulness for fine mouth-motion evidence remains
unmeasured. No visual suitability conclusion is made.

## Annotation check and evaluation quarantine

The official meeting-information files give four participants per meeting.
For ES2002 and ES2011, the official [signals table](https://groups.inf.ed.ac.uk/ami/corpus/signals.shtml)
maps the four participant roles to close-up camera names and channels.
However, the low-size mirror's `speakersegAll.xml` is not yet validated as the
ground-truth target:
its intervals carry channel and time attributes but empty `textlabel` fields in
the checked ES2002a and ES2011a files. ES2011a additionally contains channel 4,
although the mirror lists `speakersegChan0.xml` through `speakersegChan3.xml`;
the meaning of channel 4 is unknown. ES2011a's browsable directory does not
contain the manual-transcript HTML files seen for ES2002a, despite the official
availability table marking manual transcripts for all 60 scenario submeetings.
PR #19 separately records selective extraction from the official
manual-annotation archive for its train/validation cohort; use that pinned
extraction and its receipts rather than infer that validation labels are
absent. Resolve whether the low-size mirror's `speakersegAll.xml` is equivalent
before using it for targets.

During source-schema inspection, the ES2014a `speakersegAll.xml` evaluation
annotation was fetched into memory and parsed (49,529 UTF-8-reencoded bytes;
496 interval elements). No model prediction, metric, or selection used it, and
no file was saved. Nevertheless, quarantine **the entire ES2014 meeting group**
from any future sealed/locked evaluation. Keep ES2004 as the only currently
unexposed group in the official scenario evaluation partition. Do not read
additional evaluation annotations during development.

The original AMI screen and source/license statements remain at
[`R2_AMI_ACTIVE_SPEAKER_SOURCE_SCREEN_20261029.md`](R2_AMI_ACTIVE_SPEAKER_SOURCE_SCREEN_20261029.md).
The official source pages describe the released signals, transcripts, and some
annotations as CC BY 4.0; this is source evidence, not a legal review of
individual recordings or blanket product-use approval.

## Resource state and next gate

At 09:13 JST, C: had 427,327,488 bytes free and Orbitune PID 22136 was still
writing its existing run, at log step 513,350; the RTX 3080 Laptop GPU reported
15,821 / 16,384 MiB used. No process, checkpoint, or existing artifact was
changed. The 98.3 MB three-submeeting estimate is only a small source smoke and
does not include the complete train/validation data, annotations, extracted
features, or model outputs. Do not fetch it while the established 2 GiB free
space reserve is unmet.

Before admission, resolve the channel/annotation semantics, confirm the exact
manual-transcript route for held-out development meetings, and test timestamp
alignment across all required streams. Then establish enough independent
meeting groups and data storage for a predeclared task with audio-only,
video-only, joint, and mismatched-input controls. The exposed ES2014 group is
not eligible for its final locked evaluation. No training, model-quality
evaluation, sealed-audit access, or GPU job was performed here.

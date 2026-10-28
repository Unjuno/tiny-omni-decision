# AMI active-speaker source screen — 2026-10-29

## Decision

`REVIEW_NOT_ADMITTED`. The AMI Meeting Corpus is a promising synchronized
audio/video source for a concrete decision task: given a meeting timestamp and
candidate participants, identify which visible participant is speaking.
Official metadata describes synchronized close-talking and far-field
microphones with individual and room-view cameras, and says released signals,
transcriptions, and some annotations are CC BY 4.0. This is a metadata-only
screen. No media or annotations were downloaded, and no claim is made that
audio plus video is already necessary for this task.

## Evidence and task fit

The [official corpus page](https://groups.inf.ed.ac.uk/ami/corpus/) describes
100 hours of multimodal meetings. The [official annotation overview](https://groups.inf.ed.ac.uk/ami/corpus/annotation.shtml)
describes speaker-intent dialogue-act labels and video annotations such as
participant presence, face/mouth locations, head position, and hand location
for selected portions. The [official transcription page](https://groups.inf.ed.ac.uk/ami/corpus/transcription.shtml)
states that word-level transcripts are available per meeting. These ingredients
could support an active-speaker candidate-choice task grounded in synchronized
audio, video, and speaker/time annotations. The official [download page](https://groups.inf.ed.ac.uk/ami/download/)
offers selectable signals and 22 MB manual / 68 MB automatic annotation
archives; it says full-size video requires a separate distribution. The
[official signal description](https://groups.inf.ed.ac.uk/ami/corpus/signals.shtml)
reports low-size DivX video averaging 400 MB per meeting, and synchronized
50-kbit/s RealMedia video plus RealMedia audio averaging 40 MB and 2 MB per
meeting, respectively. These are published averages, not exact sizes for a
selected meeting; checksums and availability for a chosen subset remain
unknown. The RealMedia stream may be too low-bitrate for reliable visual
speaker cues, so it cannot be assumed suitable before a local decode/quality
smoke.

The official [split page](https://groups.inf.ed.ac.uk/ami/corpus/datasets.shtml)
defines training, development, and unseen evaluation partitions by meeting
sets and notes that at least one meeting is omitted from a partition because
its speakers overlap other sets. It does not establish that all proposed
train/development/evaluation meeting sets across the entire corpus are
speaker-disjoint. However, a metadata comparison of the official scenario-only
partition and the participant rosters on the official [signals page](https://groups.inf.ed.ac.uk/ami/corpus/signals.shtml)
does verify the 15 ES scenario groups specifically: SA has ES2002, ES2005,
ES2006, ES2007, ES2008, ES2009, ES2010, ES2012, ES2013, ES2015, and ES2016;
SB has ES2003 and ES2011; SC has ES2004 and ES2014. The 60 listed participant
IDs are unique across these 15 groups (zero cross-group duplicate IDs), and
the group ID keeps each set's a–d submeetings together. Thus this ES-only
partition clears the speaker-identity overlap gate at metadata level. This
does not clear other corpus partitions or media/question-content overlap.
The captured participant roster and normalized split are stored in the
[machine-readable roster manifest](AMI_ES_SCENARIO_ROSTER_20261029.json);
the manifest records the SHA-256 of the retrieved UTF-8 signals-page HTML.
Annotation availability is meeting-dependent, and the download and
synchronization path for selected videos still needs an exact inventory before
media is fetched.

If these gates pass, preflight should first inspect only small metadata and
annotation files, enumerate meetings with both required annotations and
selectable low-resolution video, verify speaker-disjoint groups across the
official partitions, and estimate exact media bytes. A task candidate is
timestamp-conditioned speaker selection over the room-view video and room
audio, with close-talk tracks used only to construct/check labels where
appropriate. Compare audio-only, video-only, joint, and mismatched-modality
controls on identical meeting/speaker groups before fitting a readout. Do not
count dialogue-act classification alone as proof that both modalities are
needed.

## Reproducibility and disposition

Metadata-only review of official AMI pages on 2026-10-29. No repository clone,
annotation archive, audio, video, training, GPU work, or sealed-audit access.
Reconsider after speaker-group split verification, annotation/video
availability inventory for the ES-only partition, exact payload-size/hash
preflight, and local storage above the established reserve. A few-meeting
RealMedia subset appears plausible by published average size but still needs
that preflight and a local video-quality smoke. The corpus license statement
is useful source evidence, but this screen is not a legal review of individual
recordings or a product-use approval.

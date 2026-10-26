# CANDOR natural-conversation AV source screen — 2026-10-26

## Disposition

**PROMISING, REVIEW ONLY; no request submitted, data acquired, or model fit.**
CANDOR is a strong task lead for a natural audio/video Decision probe because
its source reports 1,656 dyadic video-chat conversations, synchronized raw or
processed audio/video choices, and millisecond-resolution turn-taking measures.
The data are distributed only after an application is reviewed, and the original
distribution's reuse terms are not stated on the public request page. Participant
permission to share with researchers is evidence of recording disclosure, but it
does not by itself establish an ML-training license or redistribution rights.

## Pinned public-source facts

- The corpus paper reports 1,656 conversations, 1,456 unique participants, and
  850+ hours. Its collection used a third-party research ethics board; the
  consent text told participants that audio/video would be recorded and shared
  with researchers and/or made publicly available. Conversations were removed
  when an unconsented additional person appeared or recording quality failed.
- The official BetterUp research page repeats that participants consented to
  recording and researcher/public sharing. It does not state a standard open
  license for the raw recordings or a specific ML training/redistribution grant.
- The official research-data request page asks applicants for
  organizational/institutional affiliation, contact details, and intended use.
  BetterUp reviews registrations before sending short-lived download links. No
  registration was submitted.
- The public OSF replication capsule (`fbsgh`) was inspected at metadata and
  README level only. Its code capsule has an empty `data/` directory; the README
  says a reviewed “No Media Data Release” request is required even to reproduce
  analyses and describes a 2.5 GB no-media package. That README does not grant
  machine-learning training rights for the recordings. No archive, annotation
  payload, or media was downloaded or opened.
- Cornell ConvoKit says it does not separately distribute CANDOR and that the
  original distribution's license applies. Therefore a derived-tool or
  ConvoKit license must not be mistaken for a CANDOR media license.
- The source describes raw per-speaker media and processed synchronized/aligned
  media. Turn-taking annotations include speaker labels and start/stop times;
  source reports a 10 ms speech-presence time series for both participants.
  Actual timestamp conventions, audio/video alignment error, channel format,
  and completeness must be checked against an approved metadata package before
  constructing examples.

## Candidate Decision task (not yet admitted)

Test whether a frozen compact representation supports a useful, text-free
audio/video decision on conversation dynamics. A candidate task is to select
which of the two opaque participant options will take the next conversational
turn within a preregistered horizon, using only a recent causal audio/video
window. Derive the target from turn timestamps, explicitly handle overlaps and
no-next-turn cases, and do not feed transcripts, speaker IDs, filenames, or
turn-label metadata to the model. This is a task hypothesis only; the actual
annotation schema and whether both streams add information remain unverified.

If access and rights are approved, preflight in this order:

1. Read only the approved data dictionary and split/identity metadata first;
   record exact terms, revisions, hashes, channel schemas, timestamp semantics,
   and excluded recordings.
2. Build leakage groups from the connected components of the participant–
   conversation graph. Do not split a participant or conversation across
   train/validation/audit. If the graph leaves too few independent components,
   stop rather than split connected participants.
3. Confirm a causal, synchronized AV window and valid turn labels. Check whether
   participant-specific audio channels make the answer trivially available from
   audio alone; candidate option IDs must be randomly permuted opaque IDs.
4. Measure option/metadata-only predictability before fitting. Then compare
   matched audio-only, video-only, and joint frozen-feature heads on the same
   participant-disjoint rows, with timing shift, missing-stream, and
   speaker/stream mismatch controls where labels remain valid.
5. Report group-level effective sample size and clustered Accuracy, balanced
   Accuracy, NLL, Brier, ECE, and abstention coverage. Keep any supplied test
   labels sealed and do not create a blind-audit claim from an exposed split.

This is a natural conversation/turn-taking component task, not a general emotion,
social-trait, language, or Omni-quality benchmark. The sample count does not
guarantee enough independent participant groups. Do not claim that successful
turn prediction demonstrates 95–98% multimodal Decision quality.

## Sources

- [BetterUp CANDOR research page and participant-sharing statement](https://www.betterup.com/research/candor-research)
- [BetterUp research-data request and review process](https://betterup-data-requests.herokuapp.com/)
- [Corpus paper (Science Advances / PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10065445/)
- [CANDOR OSF project and public replication README](https://osf.io/fbsgh/)
- [CANDOR Corpus documentation (Cornell ConvoKit; original distribution terms apply)](https://convokit.cornell.edu/documentation/candor.html)
- [CANDOR co-author's dataset page describing available synchronized media](https://guscooney.com/candor-dataset/)

## Next gate

The public code capsule is insufficient for a model run. Proceed to metadata
acquisition only through the official reviewed access route if the applicant
has a truthful qualifying affiliation and the returned terms allow the
intended research. The no-media package could support a task/split/label audit,
but a model run still requires approved processed or raw media and explicit
terms permitting that use. Otherwise reject this source for this project and
continue with already admitted public-data candidates. No email, form,
download, GPU work, or existing artifact was touched in this screen.

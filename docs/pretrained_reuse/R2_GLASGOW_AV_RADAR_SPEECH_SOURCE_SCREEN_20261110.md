# Glasgow AV-radar speech dataset source screen — 2026-11-10

## Disposition

**REVIEW_NOT_ADMITTED. No archive, participant record, signal, audio, video,
or label payload was downloaded or opened.** The source is relevant to the
audio+video+sensor roadmap, but it cannot support a credible independent
speaker evaluation: it contains only four speakers. The audio/video records
are identifiable and shared under controlled access. The official repository
also reports a roughly 17 GB data file, incompatible with current local
capacity without an approved narrow subset.

## Pinned source facts

- Official University of Glasgow Enlighten Research Data record:
  [DOI 10.5525/gla.researchdata.2109](https://researchdata.gla.ac.uk/2109/).
- The repository page labels the dataset CC BY 4.0. This is metadata-level
  evidence only; the AV payload is explicitly controlled-access due to
  identifiable human data. No access request was submitted and no terms were
  accepted.
- The official description reports 800 labeled samples from four fluent but
  non-native English speakers; each repeats ten phonetically challenging
  words twenty times.
- Recordings contain synchronized 44.1 kHz audio, 1080p/30 fps video, and
  IR-UWB radar micro-Doppler. The stated research purpose is speech-recognition
  robustness under noise, occlusion, and other adverse conditions.
- The data file is approximately 17 GB and is not available for standard
  download; the page directs readers to request access for large/restricted
  files.

## R2 fit and blockers

- Four speakers are four independent groups, regardless of the 800 repeated
  word samples. That is insufficient for disjoint train/calibration/dev
  speaker splits and gives no useful uncertainty estimate for new speakers.
- The dataset description motivates AV-radar robustness but does not establish
  a frozen task where all three streams are necessary. A speech label can be
  strongly available from audio; matched adverse-condition and unimodal
  controls would need an auditable task protocol.
- Do not fit, cache, or evaluate this source for Path A. Reconsider only if the
  data owner confirms the exact product-training rights and supplies a larger
  consented speaker cohort with a valid held-out group split and approved
  subset access.
- No model, feature cache, run, or sealed audit was touched.

## Evidence

- [Official dataset record and collection description](https://researchdata.gla.ac.uk/2109/)
- DOI: `10.5525/gla.researchdata.2109`
- Screened 2026-11-10 from public metadata only; no archive hash exists because
  no archive was acquired.

# R2 C-MHAD video/IMU source screen — 2026-11-09

## Disposition

**REVIEW_NOT_ADMITTED.** C-MHAD is a task-relevant synchronized video and
inertial-sensor source, but the official material reviewed does not provide a
clear dataset-specific commercial/model-training license, and the source has
only 12 participant groups. Do not acquire its media or annotations, fit a
readout, or use it for product-quality selection on this evidence.

## Pinned metadata inspected

- The linked [UT Dallas author page](https://personal.utdallas.edu/~kehtar/C-MHAD.html)
  reports 240 two-minute videos: 120 smart-TV gesture streams and 120
  transition-movement streams, collected from 12 subjects. It describes six
  synchronized IMU channels sampled at 50 Hz, 640×480 video at 15 fps, and
  manually annotated action start/end times. Its Box share points to
  `https://utdallas.box.com/s/4zwus5h4khsxpullnm45o59xlfpba6a0`.
- The [2020 paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC7287800/) reports
  eight total hours and the same 12-subject / 240-stream collection. Its CC BY
  4.0 statement applies to the **article**. The reviewed author download page
  does not provide a separate data license or rights/consent terms for the
  videos, IMU values, or annotations. The Box landing response contained only
  a generic JavaScript prompt and no license information.
- The authors' linked [C-MHAD-Python repository](https://github.com/HaoranWeiUTD/C-MHAD-Python)
  was pinned at `cfc2c579ae7fcc8d3dbf50a1d1005fd5f2a6cf60`; the GitHub license
  endpoint returned 404. This code-license finding is separate from data rights.

The paper's sample detector reports fusion F1 of 81.8% vs video-only 77.8% and
inertial-only 60.3% for smart-TV gestures; for transitions it reports 82.3%,
81.1%, and 60.9%, respectively. Those are author-reported results from their
own pipeline, not evidence for this project's frozen encoders, option-probability
Decision model, or held-out quality. The small fusion deltas and lack of matched
option-level missing/shuffled controls leave “both inputs are necessary” open.

## R2 gate assessment

- **Task fit:** Continuous action detection provides timestamped actions that
  could be turned into supplied-option decisions, but no project task schema,
  sampling unit, or fixed candidate-set protocol has been established.
- **Groups:** All 240 streams come from 12 participants. Streams from one
  participant must remain together. The repository evaluation protocol uses
  200 independent groups as its reference for the paired-bootstrap interval;
  12 participants would be explicitly low-power and cannot support that gate.
- **Rights and privacy:** Dataset-specific use rights and video consent scope
  remain unknown. The article license cannot be substituted for a media/data
  license.
- **Integrity:** No official train/validation/evaluation split was identified
  on the author page. A future split would have to be participant-disjoint and
  would still have only 12 independent groups.

## Access and experiment record

Only metadata pages and the code repository's license metadata were inspected.
The Box share page was opened only as a landing page. No media file, annotation
payload, download archive, model, validation set, or sealed audit was opened.
No contact/request, download, training, evaluation, or GPU work occurred.

The companion manifest records the retrieval timestamps, pinned code revision,
page hashes, and unresolved gates. Reconsider only if the data owner supplies
explicit dataset/media/model-training rights and an adequately independent
source cohort becomes available; do not weaken the group gate to reuse these 12
participants.

# AMI headset audio + four-camera preflight — 2026-10-23

## Decision

The pinned AMI path can retrieve participant headset clips, resolve the
meeting-specific camera mapping, decode the audio, and decode a nearby frame
from all four cameras at the selected utterance midpoint. On this fixed,
small sample, the nearest camera PTS was at most 40 ms from the requested
midpoint, within one nominal 15-fps frame duration. This is sample-level
evidence that the streams share a similar meeting-relative time base; it is
not a synchronization guarantee for the full corpus.

A blinded train-only visual screen found at least two of twelve selected
positive camera labels whose target camera shows an empty room at all three
sampled instants. Therefore, the metadata mapping alone is not a reliable
speaker-visible label. **Do not train or report AMI speaker-binding quality
from this candidate set until a visibility gate and larger independent group
coverage are established.** No model inference, training, checkpoint
selection, quality metric, or sealed/unseen data access was performed.

## Frozen sample and provenance

The sample plan was generated before audio retrieval in
`ami-headset-av-selection-plan-attempt-13` (selection plan SHA-256
`f9a1ac9acc100986e9fa39f5d785cbfc17cac7e3712aca1cb40d50dd098ccc9c`). The
selection is deterministic: one earliest eligible metadata row for each of
four participants in each meeting, with a 1.5–6.0 s duration and a 2 s margin
before the shortest camera stream. For `ES2008b`, the endpoint is additionally
capped at 2,083 s to stay before the documented video truncation. It yields
12 train and 12 validation clips, with whole meetings separated:

| Split | Meetings | Clips |
|---|---|---:|
| Train | TS3009a, ES2010b, IS1004b | 12 |
| Validation | TS3005b, ES2008b, IS1006b | 12 |

The metadata source hash is
`3f7a18daaee4864b71279335b263df02919c5912658da97175b68ecdcb6b38e8`; the
meeting mapping hash is
`8ab6cdcf03ed863e839418f31c7380392397a2eed892d73b9bc1b438c7ec520a`. The AMI
Hugging Face repository HEAD was checked at execution and equaled pinned
revision `46f28f2503e2ec48f8867a84eef356c70476beab`. The Dataset Viewer
response includes a text field; the acquisition code did not access, save, or
log that field, and transcript text was not model input. The dataset card and
the earlier source audit report CC BY 4.0 for the selected AMI release.

The `meetings.xml` participant channel/camera mapping is used only as a
candidate-label source; its release lineage remains a limitation recorded in
the previous preflight. Four close-up RealMedia files have already been
integrity-checked for every meeting. Audio clips were returned as mono 16 kHz
`pcm_f32le` WAV, with decoded durations matching the selected metadata.

## Pairing and visibility results

For each of the 24 rows, the script verified the dataset row index,
`audio_id`, meeting, speaker, microphone, and start/end times against the
previously frozen metadata. It decoded the complete audio clip and sought the
nearest timestamp in all four camera streams at the utterance midpoint. All
24 × 4 candidate frames decoded. Across the 96 camera samples, the maximum
absolute requested-time to nearest-frame delta was 0.0400 s. This measures
the local PTS grid at the sampled midpoint; it does not test drift across an
entire meeting or verify that a person is speaking in view.

A separate train-only contact sheet used three instants per selected clip
(near start, midpoint, near end), randomized camera order independently by
row using seed 17, and hid the correct-camera key during the initial visual
inspection. After matching the key, two clear label-quality failures were
present:

| Blind row | Meeting / target participant | Mapped camera | Finding |
|---|---|---|---|
| R02 | TS3009a / MTD036ME | Closeup1 | Empty room at all three sampled instants |
| R11 | ES2010b / FEE039 | Closeup3 | Empty room at all three sampled instants |

These are two observed failures in this small, earliest-eligible sample, not
an estimate of corpus-wide label-noise prevalence. Other rows were not
formally adjudicated for mouth movement, occlusion, speaking identity, or
visibility quality. Validation video content was decoded at the midpoint for
technical frame availability but was not visually reviewed. No rows were
dropped or relabeled.

## Attempts, artifacts, and reproducibility

The first audio run (`ami-headset-av-preflight-attempt-14`) downloaded and
decoded 10 clips before the Dataset Viewer returned HTTP 502 for row 70,730.
That incomplete output was preserved unchanged. The script was updated to use
bounded GET-only retries for transient HTTP statuses; the frozen plan was
reused, and a fresh output was created at
`C:\CodexArtifacts\pretrained-reuse-r2\ami-headset-av-preflight-attempt-15`.

The second attempt completed 24/24 rows with 3,871,472 audio bytes. Its
`av-pair-receipt.json` SHA-256 is
`649e07132e8f7137638a014eb85f7983dc007ddf099824a9fda2453fb5fccdb4`.
Every saved clip's byte count and SHA-256 was rechecked against this receipt;
24 files were present and no partial files remained.

The blind train contact sheet and its separate camera key are outside Git:

- `C:\CodexArtifacts\pretrained-reuse-r2\ami-train-visibility-review-attempt-16\train-blind-contact-sheet.png`
  — SHA-256 `d746a7a0ba823c56d1315149c561cfef3a9f006c12717d0f68efbe7f5c43a7c1`.
- `C:\CodexArtifacts\pretrained-reuse-r2\ami-train-visibility-review-attempt-16\blind-key.json`
  — SHA-256 `a9b512d5464775fdd49f9887e23c20a92b351ed7ac182a1b94796942e1d9c611`.

The acquisition and PTS checks used Python 3.11.9, PyAV 18.1.0, the already
installed `requests`, and the existing local CPU. No dependency upgrade, GPU,
cloud compute, or paid service was used. The supporting command and deterministic
selection are implemented in `scripts/preflight_ami_headset_av_pairs.py`.

## Next gates

1. Define a pre-registered visibility/adjudication rule and apply it to a
   larger train sample without changing labels based on model outcomes.
2. Freeze more independent meeting families for validation; three meetings
   per split and twelve validation clips are far below a strong quality gate.
3. Check audio/video timing across multiple points for each candidate, not
   only one midpoint, and exclude or flag any interval outside the common
   media range. Treat TNO synchronization separately from E/I.
4. Keep audio-only, video-only, shuffled-camera, time-shift, missing-stream,
   and option-order controls in any future evaluation.

Until these gates are met, this sample supports only an acquisition/decoding
smoke and a finding that the current mapping can produce false visible-speaker
positives. It does not support training a decision model on this set or a
general audiovisual capability claim.

# Greatest Hits: fixed TRAIN source review and task disposition

## Decision

**Admit a scoped exploratory native-field development comparison. Do not admit
this source as a certified both-required, contradiction-resolution or precise
AV-synchronization benchmark.** Its questions can measure whether joint inputs
improve material/action/reaction prediction over query-only and unimodal controls.
They cannot establish that both inputs are logically necessary for the answer.
This limited admission does not satisfy the stronger data/quality gates in the
[conditional AV design](../superpowers/specs/2026-10-17-conditional-av-interaction-design.md).

All frozen labels, splits, raw media, historical preprocessing, encoders and
checkpoints remain unchanged. No sample was replaced, excluded or relabeled after
inspection. No model was loaded or trained during this review. The experimental
gate remains untrained; its 197,097,949 total parameters fit the 200M constraint,
but useful fusion and the four-modality product quality remain unproven.

## Primary-source evidence

The [author's release page](https://andrewowens.com/vis/) links the dataset and
CC BY 4.0 license. It does not currently link an official parser. Our released
three-field interpretation is consistent with observed values and the paper;
an author-provided file-schema guarantee has not been obtained.

The paper describes workers labeling **silent videos**, with cardboard merged
into paper, concrete into rock, cushion into cloth, and rubber into plastic.
Its 87.6% fine-grained material agreement is neither our label accuracy nor a
ceiling for our merged tasks. Source onsets are detected from denoised audio
peaks; the paper's allowance for small prediction shifts does not calibrate the
physical AV offset of these files. Actor/site identifiers sufficient to certify
our cross-date independence were not found. These limitations follow from the
[paper, sections 3, A1.1 and A2](https://arxiv.org/html/1512.08512).

Our inference: native labels alone do not certify a both-required task. This is
a restriction on what this experiment can claim, not evidence that useful
cross-modal information is absent.

## What actually ran

The [protocol](R2_GREATEST_HITS_SOURCE_REVIEW_PROTOCOL_20261017.md) and extractor
were committed before decoding at
`2fdba7adba6777d08137102a72abc8a76b35578d`.

- Select the minimum SHA256(`17:` + parent ID) within each of 21 TRAIN dates.
- For each selected parent, retain its earliest complete annotation; use row 0
  if none is complete. This rule does not estimate representative label accuracy.
- Persist exact IDs/rows before decoding. All 21 selected rows happened to be
  complete: 20 hit / 1 scratch; 15 static / 4 deform / 2 scatter; 8 material labels.
- Decode the native half-second mono 96 kHz PCM and every actual video frame in
  that interval. Each event contains 48,000 samples and 15 frames. Save PCM, RGB,
  PTS and a chronological contact sheet: 84 payloads, 315 frames total.
- Visually inspect all 21 contact sheets. This is a Codex qualitative inspection,
  **not an independent blinded human label audit**. Audio was analyzed as samples
  and RMS bins, not listened to. No model-quality score was computed.

Execution completed in **6.877 seconds** of extractor wall time, peak process
working set **130,727,936 bytes**. This excludes this subsequent qualitative
review, publication and tests. Existing Python 3.11.16 / NumPy 1.26.4 / PyAV
18.1.0 CPU environment; no upgrade, GPU job or cloud cost. All source pins matched
before and after extraction. Host is the existing i7-12700H / RTX 3080 Laptop
16 GiB machine; the GPU was idle at publication preparation.

```text
# Existing VideoPrism venv; CUDA_VISIBLE_DEVICES='', repo/src + repo on PYTHONPATH
C:/CodexArtifacts/pretrained-reuse-r1/videoprism-base-r1/venv/Scripts/python.exe -m scripts.inspect_greatest_hits_source_review --output <FRESH_DIRECTORY>
```

Original execution used the absolute script path (the complete argument array is
in the receipt and manifest). Start: `2026-10-16T17:58:54.971639+00:00`.
End: `2026-10-16T17:59:01.820764+00:00`. Successful exit; no live job remains.
Validation media and annotation members, official TEST media/annotations, sealed
audit and calibration payloads were not opened by this review. Split manifests
and earlier aggregate decode receipts were read for TRAIN isolation.

## Qualitative findings, including ambiguous cases

Numbers refer to the fixed chronological contact-sheet order. All exact IDs,
onsets, released labels, observations and uncertainty are in the
[small manifest](../../manifests/pretrained_reuse/greatest-hits-source-review-20261017.json).
These observations are not replacement labels or correctness percentages.

| # | Released labels: material / action / reaction | Observation and limitation |
|---|---|---|
| 0 | wood / hit / static | Wood-like trim struck; no obvious motion; composition uncertain. |
| 1 | paper / hit / deform | Notebook/paper stack struck; cover identity and subtle deformation unresolved. |
| 2 | paper / hit / static | Cardboard box struck; consistent with the source's merged paper class. |
| 3 | wood / hit / static | Light table struck near its left edge; veneer and crop-boundary ambiguity. |
| 4 | metal / hit / static | Painted rail struck; metal plausible beneath coating; approach partly at left edge. |
| 5 | cloth / hit / deform | Upholstered cushion struck with slight compression; cover composition uncertain. |
| 6 | cloth / hit / static | Red seat struck on far right; desk phone/table dominate the center. |
| 7 | metal / hit / static | Filing cabinet struck; no obvious displacement. |
| 8 | metal / hit / static | Curved metallic-looking panel struck; no obvious reaction. |
| 9 | plastic-bag / hit / deform | Bag of shredded paper indents; bag and contents have different materials. |
| 10 | tile / hit / static | Yellow hard floor struck; exact floor material uncertain. |
| 11 | rock / hit / static | Smooth pale hard floor; concrete/other finish ambiguous; contact near right crop edge. |
| 12 | tile / hit / static | Floor struck near doorway/threshold; lower-edge contact. |
| 13 | wood / hit / scatter | Strike near tree base and litter; scatter not clearly resolved in the stills. |
| 14 | rock / hit / static | Stone/concrete-looking ledge struck; no obvious motion. |
| 15 | rock / hit / static | Red masonry wall contacted; brick/mortar versus released rock label is ambiguous. |
| 16 | leaf / hit / scatter | Leaves/twigs visibly move; mixed ground materials remain. |
| 17 | rock / scratch / static | Sweep along curb beside water/leaves; material and timing ambiguous. |
| 18 | metal / hit / static | Handle is struck while wood-like door dominates the view: localization matters. |
| 19 | cloth / hit / deform | Upholstered seat struck; small deformation difficult to resolve. |
| 20 | wood / hit / static | Tree trunk struck; no obvious trunk motion. |

### Timing

The strongest 10 ms RMS-bin centers are about +5 ms from the released onset in
18 examples, +15 ms in two, and +135 ms in the scratch example (#17). Those are
**within-audio measurements**, not physical sound/contact offsets. Onsets were
already selected from audio. Contact-sheet motion provides only coarse visual
evidence, limited by blur, occlusion, about 33.37 ms frame spacing and event type.
We did not assign a physical offset or retime any file. No near-perfect
synchronization claim follows from centered peaks or shared container timestamps.

### Geometry and labels

The encoder probe's historical center crop retains 288 of 513 resized columns,
roughly **56.14% of image width** for these 456x256 inputs. The red-chair strike
(#6) is particularly vulnerable to losing contact evidence, while the phone
remains. The door-handle event (#18) shows why scene classification is insufficient.
Do not attribute failures on such inputs to fusion capacity without measuring
the actual retained view. We did not measure a corpus-wide contact-loss rate.

Some reactions need more visible detail or duration than these 0.5-second still
montages provide. Do not treat an unconfirmed reaction as a verified wrong label.
The cohort contains only one scratch and does not cover all 17 material or all
5 reaction classes. It certifies neither full annotation quality nor all-class
coverage. Date grouping and zero exact media-hash overlap remain useful, but do
not prove actor, site or near-duplicate independence.

## Admitted scope and intervention contract

| Evaluation | Status and interpretation |
|---|---|
| Frozen native 17/2/5-way fields with original answers | Scoped development comparison; report per-field/per-class and parent/date uncertainty. |
| Same events, query-only / audio-only / video-only / paired inputs | Required for measuring incremental joint benefit; no assumption of both-required truth. |
| Omit a stream without modifying the event | Original event label retained; reduced-information accuracy and calibration, not proof of recoverability. |
| Corrupt a stream in a way that may erase evidence | Robustness/sensitivity with explicit information loss; not automatic claim that the label remains inferable. |
| Replace one stream with unrelated audio/video | Sensitivity only unless valid new truth is independently established; do not score the old label as counterfactual correctness. |
| Conflict resolution / precise lag detection / both-required semantic questions | NOT_ADMITTED from the current native labels. Need separately established task truth. |
| Product quality, unseen actors/sites, four-modality 95–98% | Not established by this source review or a future single-source comparison. |

Never maximize all swap sensitivity: an irrelevant replacement should preserve
the decision. Correct paired interventions, joint gain over unimodal controls,
non-joint retention and deployment measurements are still needed for adoption.
The gate scales only the added interaction residual; it is not global modality
importance, and a large gate value is not evidence of useful information use.

## Next executable action

Before hours of caching, run the already implemented **full-extent preprocessing**
through frozen VideoPrism on this same fixed 21-event TRAIN cohort in a separate
output directory. Keep the same half-second windows, eight actual sampled frames,
weights, dtype and time coordinates. Save prepared inputs/features, verify fresh-
process replay and measure cost. Inspect the known edge-contact cases under that
single fixed policy; do not search geometries using validation scores. Full extent
reduces object scale and adds padding; it is not an assumed quality improvement.

If this input check passes, cache the immutable native-task development corpus and
freeze a learning protocol for the original Attention architecture and the four
exchange/gate arms, plus unimodal/query-only controls. Every compared arm must use
the **same new features**; comparing a new full-extent gate against an old cropped
Attention result would confound geometry with fusion. Preserve all historical
results. Measure throughput before fixing an adequate common budget, save full
learning curves/best/final and retain the four-modality regression panel. The
strict complementary-task gate remains open independently of this exploratory fit.

## Artifacts and verification

Local root: `C:/CodexArtifacts/pretrained-reuse-r2/`.

- `greatest-hits-source-review-01/`: fixed plan, raw arrays, 21 contact sheets,
  extraction receipt with source pins and all payload hashes.
- `greatest-hits-source-review-verification-01/`: RED/GREEN logs, execution log,
  qualitative review, publication/audit scripts, preservation receipt and CPU/CI
  verification. No large payload or media is added to Git.

Plan SHA256:
`951d1078d3fae228256bb398bd636fea2393fd7aa4b05969ef04655c7f475ae5`.
Receipt/review/output hashes are recorded in the public manifest. The extractor
has six new tests for deterministic date coverage, earliest-complete/fallback
selection, acoustic-only RMS interpretation and invalid audio rejection. Full
CPU verification: **2,248 passed / 1 skipped in 31.57 seconds**; Ruff PASS;
all 25 existing manifest commands PASS. The new research manifest is independently
checked against the extraction/review receipts, fixed IDs, arrays, RMS values and
hashes; it is not one of those older CLI manifest formats. RED six-failure and
GREEN six-pass logs preceded the committed extraction.

Independent publication audit PASS: all **8,350** previously protected files
(994 old-model, 7,310 earlier source/input, 46 encoder files) retain their hashes.
All 84 new payloads and three receipts are verified. The audit independently
recomputes the date selection and acoustic bins; it cannot certify qualitative
judgments. Source/review verification files:

- Audit source SHA256:
  `90d694bb510c0f24477a517fbb06db92af78fb61d30e6d49dc138ae867fa2f39`.
- Qualitative review SHA256:
  `207a0463068ae47ebd20a2d1154d839f395a9b8f1adb979fa08015feba0e9732`.
- Public manifest SHA256:
  `553daf8dc87d78957daf0ddc6bcb28b850b20e770fa03149c86f05e9b35a948a`.

The publication-head CI receipt is saved at `publication/ci.json` under the
verification directory after push. Earlier-head success is not substituted.

The product goal remains active and incomplete. Gate quality, physical sync,
actor/site generalization, valid complementary interventions and 95–98% accuracy
have not been established. No new quality training, calibration or quantization
was performed here.

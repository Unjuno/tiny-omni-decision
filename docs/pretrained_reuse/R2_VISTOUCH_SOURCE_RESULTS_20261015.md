# VisTouch public contact-material source — complete metadata audit

**Explicit source terms; metadata PASS; native training NOT_ADMITTED.** Selected
use case: supplied-option material recognition from visual contact observations,
contact audio and force traces. This domain candidate supports studying whether
multiple native inputs help; it does not establish that they do, or replace the
full general <=200M/four-modality95–98% goal.

## Immutable source and permissions

Repository revision `liangnjupt/VisTouch@a569c7e572a78f232e819b652e0ebb718952e476`,
tree `3d752cff8c4b54c8d78af0b4648f9749109a9036`. Primary
[DATA_LICENSE](https://github.com/liangnjupt/VisTouch/blob/a569c7e572a78f232e819b652e0ebb718952e476/DATA_LICENSE)
explicitly grants CC BY4.0 to audio/tactile/video/metadata; MIT code is separate.
Attribute the dataset and mark preprocessing changes if later used. This screen
downloads only12 immutable license/metadata/documentation/source-text files,
**3,339,524 bytes**, verifies exact Git blob SHA1/size plus SHA256, and executes
none of the upstream code. No media, upstream model weights or logs were fetched.
The complete12-file set is not the complete upstream repository/frame index.

Source root `C:/CodexArtifacts/pretrained-reuse-r2/vistouch-source-screen-01`;
receipt SHA `85b86f9b50e8067408db5e70ef326f92e238c41932a004fee12c7f0a3d6dce7e`.
The bounded text fetcher/plan remain archived in the same artifact parent,
alongside their hashes; actual retrieval5.0567s. The dataset authors' original
citation is preserved in fetched `CITATION.cff`; no source-code license is inferred
from the data grant.

## Actual full-index accounting

[Prospective contract](R2_VISTOUCH_SOURCE_PROTOCOL_20261015.md), source commit
`1437710a291985185d2c5a502bdf9bf4d5ad736b`. All classes/sessions/samples/clips_index
rows were checked, including native parent/class/force/split/interval/duration
consistency; no row was removed or new split invented.

| Quantity | Actual metadata |
|---|---:|
| Material classes |8: brass, linen, paper, polyester, silk, spandex, stone, wood |
| Original source sessions |24 |
| Listed source audio/video/tactile assets |72; physical files not verified |
| Parent samples |312 (208 train /104 test) |
| Materialized microclips |10,498 (7,019 train /3,479 test) |
| Source sessions by split |16 train /8 test |
| Nominal force conditions |Train3/6, test9 (source force_n metadata) |
| Session/raw-source-filename aliases across split |0 in supplied metadata |
| Independent validation |Absent |
| Unique source-window keys /exact duplicates |10,498 /0 |
| Union of covered source windows |1,683.9s (~28.07min) |
| Sum of represented window durations |5,249.0s; overlapping coverage3.11717× |
| Stored playback duration sum |20,996s (~5.83h), after4×stretch |

The microclip tier extracts overlapping0.5s windows at0.15s stride and stretches
each to2s. Thus10,498 clips and5.83h playback are not10,498 independent episodes
or5.83h of unique native observation. An independent rational-arithmetic event
sweep, using no metadata-auditor helper, reproduces the full source-time union,
split counts, all window keys/duplicate counts and duration sums; simultaneous
window multiplicity peaks at4. All original rows remain visible.

Both split class counts and force/source groups are in the
[small exact result](R2_VISTOUCH_SOURCE_RESULTS_20261015.json). Metadata filenames
are provenance/labels only; none is an admitted learned input. The nominal force
field does not establish calibrated physical sensor values or a safety policy.
Raw media hash and normalized-content overlap remain **UNVERIFIED**, and physical
specimen identity is **UNKNOWN**. Separate session IDs alone are not independent
validation or object-held-out generalization.

## Timing evidence and boundary uncertainty

The source README describes synchronized signals, but its
[alignment report](https://github.com/liangnjupt/VisTouch/blob/a569c7e572a78f232e819b652e0ebb718952e476/docs/alignment_report.md)
states video epochs are linearly mapped onto the audio interval and sensor
subsecond timestamps are interpolated from integer-second records. Those are
documented approximations, not measured frame-level synchronization. Video also
has a burned-in timestamp overlay; its region and safe model preprocessing have
not been inspected in actual media. Preserve this possible source cue as a gate.

The numeric audit flags **124 windows** beyond the *declared integer* audio/
tactile boundary, with maximum0.981s. All discrepancies are within one second.
This does **not** establish124 physically invalid clips: a final tactile sample
could lie later within the recorded integer second. Full discrepancy IDs are
retained in the raw audit. Do not delete these windows, infer exact last-sample
time, or change timing to improve scores. They are unresolved boundary checks,
not a claim that the release violates its actual sensor coverage.

## Execution, reload/accounting and access

```powershell
python -m scripts.audit_vistouch_metadata --output C:/CodexArtifacts/pretrained-reuse-r2/vistouch-metadata-audit-01.json
python C:/CodexArtifacts/pretrained-reuse-r2/vistouch-independent-metadata-01.py
```

Main audit0.29577s, peak70,057,984 bytes,21 pins; no models/GPU. Prepared accounting
is read back, sources rehashed, CPU/time/disk checked and terminal bytes reserved
inside the fixed1MiB combined report cap before PASS publication. Independent
sweep0.39552s, with23 pinned sources/products/consumer; it validates the numeric
accounting, not pixels, timing accuracy or material recognition.

Main result SHA `8d53c11d3b3d9ee76d41d4f806beda1080947f9c3eba49a7eae7502dd625f8e5`;
the prepared report and independent result hashes are in the small summary.
Artifacts remain outside Git; no existing dataset/config/checkpoint was changed.

Actual native archive access remains **UNVERIFIED**. The publisher links Baidu
Netdisk; anonymous HTTP reaches its extraction-code page, but no archive listing
or payload was retrieved. The browser tool fails before control with
`failed to write kernel assets ... (os error 3)`. The HTTP receipt was saved, but
printing Chinese signal keys caused a cp932 stdout encoding error; this is not
evidence of a download failure. Exact access/limitation receipts are
`vistouch-public-access-01.json` and `vistouch-public-access-limitations-01.json`
under `C:/CodexArtifacts/pretrained-reuse-r2`. No account, client installation,
credentials, cloud compute or media transfer was attempted. A working supported
browser or publisher-provided direct native archive listing is needed for the
next payload gate; no scheduled/finished native experiment is claimed.

## Next gate and remaining product scope

If native sources become accessible, use the original unaltered source tier,
not stretched clips as new native temporal observations. Verify actual media
hashes/content/specimen grouping, units, capture/export bounds and label-free
preprocessing before a frozen split/validation protocol and fitting. Keep all8
classes and grouped sessions, length/no-input controls and matched unimodal/joint
ablations. Independent validation and class/domain coverage must be specified
before checkpoint selection; no tuning on supplied test metrics or legacy audit.

No Accuracy/NLL/Brier/ECE or joint benefit was measured. This source does not
close temporal semantics, native audio/video/sensor Decision quality, text
conditions, OOD or complete Omni package/residency requirements. The previous
negative and interrupted experiments remain protected; no product promotion,
Teacher training, compression, cloud charge or legacy GPU restart.

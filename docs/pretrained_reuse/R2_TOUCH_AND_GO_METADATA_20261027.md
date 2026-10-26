# Touch and Go: complete label-snapshot feasibility audit

**Not admitted for a full 20-material quality fit.** The complete official
annotation file contains 250,169 frame labels from only 138 recording names and
24 date prefixes. Natural Fabric appears in one recording/31 frames. It cannot
span independent training and evaluation groups. Reusing neighboring frames or
silently dropping this category would not solve that limitation.

This is a new source/task feasibility result, not a model experiment. No raw
RGB/GelSight video, feature encoder, fitted head, calibration, validation score
or sealed audit was used. The four-modality and joint-input quality targets
remain unmet.

## Authoritative sources actually read

The [official project page](https://touch-and-go.github.io/) describes paired
egocentric RGB and GelSight recordings, material labels and touch onsets. It
identifies `time1.npy` and `time2.npy` as the recording times for the respective
video frames and links the [official public Drive release](https://drive.google.com/drive/folders/1NDasyshDCL9aaQzxjn_-Q5MBURRT360B).
The page carries a CC BY 4.0 declaration. This is publisher evidence, not an
independent check of asset ownership. No separate code license was established
and no upstream code was imported or executed. Material labels do not themselves
prove a decision needs both vision and touch; native audio is not established.

Drive metadata search returned no children, but the connector's folder `fetch`
correctly listed the public release. Browser readback independently displayed
the top-level files. The connected provider returned a 100-folder page for the
dataset directory; that page is explicitly **partial**, not a complete corpus
count. The 138 recording count below comes from the **whole label file**, not
that folder page. Provider created/modified times are retained in private
metadata receipts. The Google Drive release has no immutable revision pinned
here; hashes bind exactly the observed bytes, not a publisher-authenticated
immutable version.

The official [label file](https://drive.google.com/file/d/1YMK2HfyVzl9ptc5t-A7l2PP-fLhB47kp/view)
is 8,369,167 bytes, SHA-256
`36f358e2f0ec36a9909a51d593a3d086c605203859fd5eec0c766d00200afc9a`.
The [category reference](https://drive.google.com/file/d/18TkSzfxyhjVGHZ0cgE3hF_0DoFy9ivRp/view)
is 275 bytes, SHA-256
`84f6c3a027cc962a2eee786b9776546d56a9b8edacbaa766f52efd401b7719aa`.
The complete `-1..19` category mapping is retained. Upstream spellings, including
`Ruber` and apostrophes in two names, are not silently changed.

## Complete annotation result

| Quantity | Observed |
|---|---:|
| Frame-label rows / unique recording-frame identities | 250,169 / 250,169 |
| Duplicate or conflicting frame identities | 0 |
| Distinct recording names | 138 |
| Distinct date prefixes | 24 |
| Material-labeled frames, classes 0–19 | 167,428 |
| Upstream Inconclusive (`-1`) frames | 82,741 |
| Natural Fabric | 31 frames / 1 recording / 1 date |
| Ruber (upstream spelling) | 540 frames / 8 recordings / 4 dates |
| Sand | 272 frames / 8 recordings / 7 dates |
| Others | 792 frames / 6 recordings / 5 dates |

All per-class frame/recording/date counts are in the
[machine-readable audit](R2_TOUCH_AND_GO_METADATA_20261027.json). There is no
class reduction, sampling, task relabeling or model selection. `-1` is retained
in accounting; its upstream note says not to use it for material classification.

A recording name and its date prefix are not subject IDs or unique object IDs.
Independence across people, objects, sites and collection sessions is unverified.
Even treating every recording as independent would leave fewer than the
roadmap's provisional 200-group quality screen, before any split. A full
train/calibration/validation/audit design with each material represented is
impossible at recording granularity for the observed rare class. Framewise
random splitting would conceal this issue by sharing the same recordings.

## Bounded timestamp check, no video acquisition

Before opening timestamp arrays, select the first recording returned in the
observed public folder page: `20220319_001819`. Its metadata lists two MP4s and
two NPY timestamp files. Only `time1.npy` and `time2.npy` were fetched, 2,664
bytes each; they load with `allow_pickle=False` to finite, strictly increasing
317-element float64 arrays.

| Numeric timestamp diagnostic | RGB array | GelSight array |
|---|---:|---:|
| Span in native units | 10.69325447 | 10.67763448 |
| Median step in native units | 0.03223288 | 0.03220904 |

Maximum same-index absolute timestamp difference is `0.0157864094` native units.
These magnitudes are compatible with seconds-based recording times, but unit
semantics and physical capture synchronization were not independently verified.
No videos were decoded, so timestamp/video frame-count agreement is untested.
Equal timestamp counts cannot certify simultaneous exposure or contact timing.

Timestamp hashes:
`c9d9db4d45ce77f7ace9f9a0ee69853c6fd6a17d01e2cf14693e45709b035159`
and `d27b9430199fa2b481a0d2b3c04a5a25fad5f44e1d48fcc813f26ce91b847469`.
Selected annotation/reference/timestamp files total 8,374,770 retained bytes;
HTML/folder metadata are additional small receipts. This is not a measured
total network-byte count.

## Disposition and reproduction

Preserve this source as a possible low-power vision/tactile research lead.
Do not launch the full material-classification fit, call 250k correlated frames
250k independent examples, or advertise product quality from it. Additional
independent recordings for the rare classes and source-group identities would
be needed, along with a prospective task whose labels and controls establish
joint-input benefit. No large acquisition follows this failed feasibility gate.
Revisit only if source coverage/provenance or the approved task scope changes.

[ObjectFolder Real](https://objectfolder.stanford.edu/) was also screened from
its official description: 100 real objects with separate modality-specific
collection setups. This does not establish synchronized audiovisual episodes
or adequate independent groups for the current task matrix. No ObjectFolder
data or code was downloaded; no license from its simulated datasets was applied
to the real recordings.

Local raw metadata:
`C:/CodexArtifacts/pretrained-reuse-r2/touch-and-go-source-audit-20261027-attempt-01/`.
The original metadata and numeric arrays remain outside Git. Legacy runs,
checkpoints, datasets and other worktrees were preserved.

```powershell
python -m scripts.audit_touch_and_go_metadata --output C:/CodexArtifacts/pretrained-reuse-r2/touch-and-go-source-audit-NEW.json
```

Authoritative audit output is `audit-result-02.json`, SHA-256
`82103e2c3204f4f4c2acc24b6d80279cea06b216b52d9d01e0c885d2c4e35eda`.
Output 01 is preserved; 02 only shortened a lint-overlong scope string.
Initial parsing exposed a trailing space in `Wood: 3 `; the parser now strips
line-edge whitespace while preserving category names and validates the full
category range. CPU regression tests cover correlated row accounting, rare
class retention, duplicate/conflicting/unsafe identities, invalid dates,
unknown categories and timestamp agreement without a synchronization claim.

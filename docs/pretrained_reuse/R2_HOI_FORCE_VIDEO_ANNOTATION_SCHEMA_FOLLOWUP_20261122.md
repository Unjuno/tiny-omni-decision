# Hoi! annotation-schema follow-up — 2026-11-22

## Result

The pinned public processing code resolves part of the label-schema gate for
Hoi!, without accessing any dataset payload. At revision
`f60ef1669859e2b3ac71743923292ccbedb0e1df`,
`interaction_splitting_verifier.py` loads temporal windows with
`start_ns`, `end_ns`, and `duration_s`. The annotation UI asks a reviewer to
confirm each window, choose `open`, `close`, or blank, and optionally enter an
integer `index`; its writer emits those same fields as `single_word_description`
and `index` in each `window_N` record. The inspected file SHA-256 is
`5ba58e74c438cf7b2e1d440f5e822a9c3e3581bfc52c63c0e0596ba0965eb23`.

This makes a binary `[open, close]` choice mapping structurally plausible for
the manually labeled interaction window. It does **not** prove that every
released row has an unambiguous verified label, that blank labels are rare,
that `index` identifies an object, or that the interval provides leakage-free
model inputs. The code and public README still do not establish stable
participant, object, or environment split identifiers. No annotation payload
was inspected to answer these questions.

## Disposition

Hoi! remains `REVIEW_NOT_ADMITTED`. The annotation code is a useful label
schema lead, but it does not close the data-access, group-split, class-support,
or task-validity gates. The full source lists 3.19 TB and its publisher page
reports a guestbook name field; no identity was supplied and no access request
or payload download was made. Do not acquire data until the access flow is
approved and a bounded, rights-compliant subset and leakage-resistant split
are verified.

## Reproducibility

- Repository: `https://github.com/timengelbracht/hoi-dataset-tools`
- Exact revision: `f60ef1669859e2b3ac71743923292ccbedb0e1df`
- Files inspected: `data_processing/src/hoi/annotation_tools/interaction_splitting_verifier.py` and `data_processing/README.md`
- Data payload / annotations / model / GPU: not accessed / not opened / not loaded / not used
- External read-only checkout: `C:/CodexLocalArtifacts/public-source-audits/hoi-dataset-tools-f60ef166`

The base screen is [here](R2_HOI_FORCE_VIDEO_SOURCE_SCREEN_20261122.md); its
machine-readable receipt is
[`hoi-force-video-source-screen-20261122.json`](../../manifests/pretrained_reuse/hoi-force-video-source-screen-20261122.json).

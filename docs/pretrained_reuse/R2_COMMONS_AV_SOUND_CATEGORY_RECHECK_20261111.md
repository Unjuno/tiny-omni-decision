# Wikimedia Commons synchronized AV sound-category recheck — 2026-11-11

## Decision

**NOT_ADMITTED for the scoped native synchronized AV sound-source task.** The
metadata-only recheck found only 34 directly listed files across the explicitly
sound-related animal categories inspected (19 animal-sound files, 11 bird-song
files, and 4 cat-sound files), before deduplication, per-file rights review, or
independent-recording grouping. This is far below the roadmap's provisional
1,000-example / 200-independent-group screen. No media or model data was
acquired, and no training or evaluation was run.

This is a disposition of the scoped category candidate, **not a claim that all
of Wikimedia Commons has been enumerated**. The remaining Commons category tree
and broad video categories have not been inventoried completely.

## Scope and method

On 2026-11-10 UTC, the official Wikimedia Commons MediaWiki API was queried
using `action=query`, `list=categorymembers`, `cmtype=file|subcat`,
`cmlimit=max`, and `format=json`, with a project-identifying User-Agent. The
queries listed category membership only; they did not request file bytes,
thumbnails, audio streams, video streams, page text, or annotation payloads.

| Category | Direct files | Child categories | Observation |
|---|---:|---:|---|
| `Videos of animals producing sounds` | 19 | 3 | The three child categories were birds singing, cat sounds, and squirrel sounds. |
| `Videos of birds singing` | 11 | 0 | Explicit bird-song category. |
| `Videos of cat sounds` | 4 | 3 | Child categories were growling cats, hissing cats, and cats purring. |
| `Videos of growling cats` | 0 | 0 | No direct files listed. |
| `Videos of hissing cats` | 0 | 0 | No direct files listed. |
| `Videos of cats purring` | 0 | 0 | No direct files listed. |
| `Videos of squirrel sounds` | 0 | 1 | Child category was Sciurus vulgaris sounds. |
| `Videos of Sciurus vulgaris sounds` | 0 | 0 | No direct files listed. |

The 34-file count is a conservative count of the direct file entries in the
four non-empty categories. It is not an eligible-example count: exact media
identity, duplicate files, source-recording identity, upload license, and
whether both streams are present were not checked. The category counts alone
cannot establish 200 independent groups.

For context, the broader `Videos of animals` category listed 120 direct files,
but those entries are not all documented as sound-bearing or synchronized and
were not counted toward this task. `Videos of birds` listed no direct files;
an API category-prefix lookup returned 43 names beginning with `Videos of
birds`, but their parent-child relationships were not verified and they were
not all enumerated. Neither fact supports admission.

## Rights, labels, groups, and split status

- Item-level licenses and attribution requirements: **not checked**.
- Exact file page IDs/revision IDs, media hashes, original recording/session
  IDs, and uploader/source group identities: **not captured**.
- Whether each listed video contains synchronized audio: **unknown**.
- Event-level labels showing which visible source produced the sound: **not
  present in category membership metadata**.
- Exposure exclusions and train/calibration/development splits: **not created**.
- Candidate task “select which visible source is sounding” therefore has no
  verified ground truth or split plan. Category labels alone are not enough to
  create those labels.

The candidate fails before any media acquisition or annotation work. Do not
derive additional windows or questions from these entries to inflate the
independent-group count. Any future Commons effort must start from a separately
defined task and a complete, rights-aware inventory with stable source-group
identity; it must not treat this partial category screen as an eligible pool.

## API integrity and limits

The first exploratory request used unsupported `cmtype=all` and returned a
MediaWiki warning with zero rows. It was discarded. Corrected requests used
`cmtype=file|subcat`. A later burst received the API's rate-limit response; the
requests were paused before one subsequent category query. No alternate access
route was used to bypass the limit. Because the Commons API is mutable and the
responses were not archived as immutable snapshots, the counts are an observed
metadata screen rather than a version-pinned corpus manifest.

The API endpoint was `https://commons.wikimedia.org/w/api.php`. No Commons media
was downloaded, no item-level license was accepted, no GPU or model was used,
and no sealed or previously held-out payload was opened.

## Follow-up disposition

Keep the joint-modality R2 gate open. Continue only if a future candidate has a
predeclared both-input task, at least 1,000 plausibly eligible examples across
200 independent original recordings/groups, and a feasible item-level rights
and grouping audit before media access. Otherwise use the already measured
Path A unimodal paths and do not fit another fusion head.

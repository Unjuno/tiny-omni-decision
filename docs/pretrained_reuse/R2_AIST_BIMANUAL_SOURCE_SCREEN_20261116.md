# AIST Bimanual Manipulation source/task screen — 2026-11-16

## Disposition

**`REVIEW_NOT_ADMITTED` for Path A/R2-M fitting or quality evaluation.** The
official page publishes a CC BY 4.0 notice and describes synchronized
multi-camera RGB/depth, robot joint state/action, and task text. The metadata
does not provide a train/validation/test split or an auditable independent
session/episode grouping. The task-list counts also conflict across official
pages, and its action description/prompt can reveal the target. No media,
HDF5, model, annotations beyond task metadata, or evaluation labels were
downloaded or opened.

## Pinned official metadata

- AIST dataset page: <https://aistairc.github.io/aist_bimanip_site/index.html>,
  accessed 2026-11-16. It reports CC BY 4.0, over 10,000 episodes, 100+ tasks,
  synchronized multi-camera views and joint tracking. The rendered task table
  contains 117 dated task rows across 61 distinct calendar dates.
- Official task CSV:
  <https://aistairc.github.io/aist_bimanip_site/statics/docs/sheet.csv>.
  HTTP 200, 28,827 UTF-8 bytes, SHA-256
  `62bd7daf3b43792575624cfc8942e32c65afcd0a1ff0f875850c3c666de10e33`.
  It has 119 task rows and its `HDF5 count` column sums to 12,025. Fields are
  task ID/name, action type/text, environment, taxonomy, skill, and HDF5 count;
  it has no split, episode identity, participant/session identity, or recording
  timestamp fields.
- Dataset details page:
  <https://aistairc.github.io/aist_bimanip_site/dataset.html>, accessed
  2026-11-16. Its summary says “117 episodes,” while describing four RGB views,
  depth, joint positions/velocities, actions, and task prompts. This conflicts
  with the main page's 10,000+ episode claim and the CSV sum above.
- The AIST public release announcement:
  <https://www.aist.go.jp/aist_j/press_release/pr2025/pr20250902/pr20250902.html>
  (2025-09-02) describes approximately 10,000 action sequences and links the
  dataset to the official page. It does not resolve the per-file or split
  counts.

The main page and details page were served with `Last-Modified: 2025-09-04`
and ETags `W/"68b9cb8f-19c94"` and `W/"68b9cb8f-6263"`, respectively. Their
UTF-8 response-body SHA-256 values on 2026-11-16 were
`0f58a3d218c83ecf5cfa96e0f7538ed536ceadf73f21fbf7d0f975d19ec09f1c` and
`1556c7df759a39efdf4ffb405c7a0a30fab4a80e47270b6e8dcdf4f6668cbd8a`.

## R2-M gate review

| Gate | Finding | Disposition |
|---|---|---|
| Product-compatible rights | Official page states CC BY 4.0 | Promising; retain attribution and verify source/version before any use |
| Paired modalities | RGB/depth, robot joint observations/actions, prompts are described | Structurally promising; no payload inspected |
| Supplied-option Decision target | Task/action text exists, but no upstream option-choice labels or frozen Decision protocol | Not established |
| Both-input necessity | A task prompt or action description may expose the target; no video-only, state-only, prompt-only, or matched controls | Not established |
| Independent split | No split or participant/session/episode identity in the task CSV; 61 dates appear in the page table, but dates are not proven session groups | Fail closed; the provisional 200-independent-validation-group floor is unverified |
| Inventory consistency | Main page says >10,000, CSV sums to 12,025, details page says 117 episodes | Conflict; clarify before using counts or estimating acquisition |
| Local feasibility | Official downloader metadata indicates 117 archive files; sizes and representative subset requirements are not verified | Unknown; do not fetch archives |

Do not create random episode-level validation from the published task counts.
Reconsider only if AIST provides a consistent inventory, a defensible
session/episode grouping and split strategy with at least the provisional
validation floor, and a target/control design showing video and state are
both needed without task-prompt leakage. This screen does not authorize
payload acquisition, fitting, validation, or promotion.

## Reproduction and boundary

- Repository/base commit for this screen: `bfb13bf6e9ac4f2ce244eb51e470afa907587380`.
- Read-only requests fetched the official HTML pages and the 28.8 KB task CSV.
  Counts were computed from the CSV `HDF5 count` field and ISO dates in the
  official main-page task table.
- No Dropbox links were followed. No archives, media, sensor/action trajectories,
  models, feature caches, validation/test labels, or sealed data were accessed.
- No model/GPU work, dataset mutation, or training was performed.

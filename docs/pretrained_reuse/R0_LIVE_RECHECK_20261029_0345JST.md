# R0 live recheck — 2026-10-29 03:45 JST

## Active roadmap branch

- Worktree branch: `docs/pretrained-reuse-roadmap-20261010`, clean at
  `34b26cc07b02a125490f325b4aebbb7e03f13a83`; fetched remote head matched.
- PR #15 remains open and Draft, with the same head and base
  `09246d7b1821662a6b1eba47fc08c43f9dc3863c`.
- Both commit checks for `34b26cc0` completed successfully (push run
  `40747355029`, PR run `40747364507`).

## Current machine resources

- RTX 3080 Laptop: 15,821 / 16,384 MiB VRAM used, 100% utilization.
- Orbitune training PIDs `31624` and `22136`, and watchdog PIDs `29152` and
  `20352`, are still present. None was signaled or changed.
- C: has 0.73 GiB free. The common Path A fit guard requires 2 GiB free on its
  output volume. No new fit or dataset download was started.

## Reconciliation of a stale legacy receipt

The preserved external run
`C:/CodexArtifacts/pretrained-reuse-r2/greatest-hits-matched-train-E0G0-01/`
contains conflicting status artifacts:

| Artifact | Evidence |
| --- | --- |
| `result.json` | SHA-256 `02a1f0658f4223442cfd08c5456a1a90ac08217258bf4553d9cf3c00a1fc8e7f`; its stale status is `RUNNING`, step 5,376, PID 19,988. |
| `safety-stop.json` | SHA-256 `3336177f51fa894b14257665dcb1b30153d684d043e08652e2010b4f109f3d3e`; status `INCOMPLETE_SAFETY_STOP_GRADIENT_ANOMALY`, last flushed step 5,385, 43,080 presentations. It records max preclip gradient norm 4,681,888.5 and repeated extreme CE values. |
| `best.json` | SHA-256 `2c40c9ebdf0ae84aaaa07c6cc87b0ecb24fd54fc4c3d3ec86017d5eb42682c02`; selected step 4,096. |
| `best.pt` | SHA-256 `7e73ca5a5cdd8efde8277e37876dd2b604aa840e0dafe6566cc3afa856f0e87f`, matching the selected checkpoint hash. |
| `training-history.jsonl` | SHA-256 `543a6f252103b27687b4c90c35b50b81968bcb92c858b27914d67a0f83b329a5`. |
| `learning-curve.json` | SHA-256 `62a4695c0502cba94ed52c7a3036189902647c19cb40f2f78c178baff2948956`. |

The safety-stop receipt's embedded result hash matches `result.json`. Its
termination entry says PID 19,988 was stopped after the gradient-anomaly review
on 2026-10-17. That PID is absent now. Best reload was **not** verified, the
32,768-step budget was **not** reached, and the receipt explicitly makes no
optimizer/scheduler resume claim. Treat this as an incomplete safety-stopped
experiment despite the stale `RUNNING` field. Preserve all files; do not resume
or relabel it as completed.

## Decision

The current device/output guards still fail for another model fit. Previously
completed Path A, GRID, Eidon, Greatest Hits and synthetic AV experiments must
not be repeated on their exposed development records. EgoViz-120 remains a
metadata-only preflight; participant independence, raw stream synchronization,
label quality, and a task requiring both video and IMU remain unverified. No
sealed audit was opened. Continue R2 source/task admission work until there is a
newly justified, rights-cleared, sufficiently grouped task and the local
resource preflight passes.


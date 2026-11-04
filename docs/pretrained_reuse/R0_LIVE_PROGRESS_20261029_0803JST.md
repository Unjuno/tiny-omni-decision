# R0 live progress recheck — 2026-10-29 08:03 JST

The read-only process handle still identifies Orbitune PID `22136`, started
2026-10-28 09:12:42 JST. Its latest log row at 08:03:18 is step `511150`,
`12,190,945,719` events, about `14,330 events/s`. The latest *saved*
`model.pt` is still the 07:49:52 checkpoint: step `510759`,
`12,180,004,749` events, SHA-256
`3534DF6DA88237CCB759129072EA4C45C9871C0B64C9D05EFB7A8A6EB5393C96`.
Its best step remains `455514` (validation loss `-1.6231351792812347`); the
latest saved validation record is step `510759`, loss `-1.4775453209877014`.
Do not confuse the moving log row with a saved checkpoint or selected best.

The configured 30M-event interval has about 19.06M events remaining from the
last saved state; the recent log rates of 13.4–14.3k events/s project a save
near 08:25 JST. This is not a process-completion estimate. The RTX 3080 Laptop
was at 15,821/16,384 MiB and 95% utilization. C: free space was 441,798,656
bytes (~0.41 GiB), below the 2-GiB reserve. No GPU fit is in progress from this
worktree; the small image-count loss comparison completed on CPU. PID 22136,
its run directory and checkpoint were not modified.

Repository branch `docs/pretrained-reuse-roadmap-20261010` was clean at
`b4f4f12d41fe706d59a15e4ef3a748d62d5f751f`; PR #15 remained Draft/Open and its
latest push/PR checks passed. No other worktree or PR was changed.

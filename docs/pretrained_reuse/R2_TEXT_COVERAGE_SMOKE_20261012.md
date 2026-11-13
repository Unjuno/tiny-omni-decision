# Broader text coverage smoke

Separate process passed2updates on16 training questions and8+8 validation
questions. Best/final weights retained; frozen weights unchanged and selected
checkpoint reload exact. Peak allocated VRAM2443544576 bytes, timed loop1.552s.
The smoke is execution evidence only; no quality or full-run throughput claim.
Local output: `C:/CodexArtifacts/pretrained-reuse-r2/text-coverage-smoke-01`.
Source tree was at8cdb80b with the new runner not yet committed; exact runner
hash and command are in the accompanying config. Full run must use committed code.
GPU after smoke:54MiB used/16122MiB free,0% utilization. Desktop Codex and
LM Studio GPU contexts were present; neither was stopped or changed.
Ruff and412CPU tests passed (one skipped). Storage constraint is now resolved.

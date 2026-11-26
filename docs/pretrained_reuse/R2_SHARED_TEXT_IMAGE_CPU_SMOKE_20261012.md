# Shared-text image comparison CPU smoke

Execution check only: eight train/eight validation images, four questions each,
one epoch per condition, seed17. This deliberately tiny smoke is not a quality
comparison or checkpoint selection for deployment. Full 24-epoch paired experiment
remains pending until the active numeric GPU training is finished and a separate
GPU smoke passes. No existing corpus, weight or running-training setting changed.

Both MiniLM and frozen Cross-Encoder conditions used exactly equal bilinear head
initial tensors and example-order hashes. Finite training, immutable source-file
hashes, split checks and exact selected-checkpoint reload all passed. Cache mean
parity versus prior global features stayed within the existing 1e-5 gate.
CPU only; CUDA metrics are null. Outcomes are retained in adjacent JSON.

The image-only pipeline counts are 40,331,392 for MiniLM and 45,655,169 for Cross-
Encoder. Sharing only saves parameters in a combined system that already needs
the SAME Cross-Encoder for text decisions; it does not shrink image-only inference.
The frozen zero numeric delta was omitted from this control checkpoint's base,
so this implementation adds no 131,712-row delta parameters. Numerical embedding
candidate selection and joint multimodal quality remain separate/open.

The runner was first executed from an uncommitted new script; its exact file hash
is saved in the smoke config and the script is included with this report. No
claim that the parent commit already contained it. Artifacts are preserved at
`C:/CodexArtifacts/pretrained-reuse-r2/shared-text-image-cpu-smoke-01`.

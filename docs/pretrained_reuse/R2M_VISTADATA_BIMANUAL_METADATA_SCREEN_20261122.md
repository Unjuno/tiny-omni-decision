# VistaData-Bimanual metadata screen — 2026-11-22

## Disposition

Keep this source `REVIEW_NOT_ADMITTED`. The pinned public release is CC BY 4.0
and contains synchronized robot video, state and one GelSight tactile stream,
but its Zarr metadata reports only 30 episode groups for a single
unscrew-coffee-jar task. The public card does not identify independent
participant/session groups, an official held-out split, or a supplied-option
Decision target requiring both video and touch. This is too narrow to support
the project's general multimodal quality claim. Do not treat episode count as
independent-group count.

No full archive was downloaded or saved. A sparse ZIP range inspection read
the central directory and Zarr episode metadata; approximately 33.0 MB of
range response bytes were transferred. Some range blocks may overlap unrelated
compressed archive members; no media frames were extracted, decoded, scored or
used. No model, GPU, training, validation or sealed audit was involved.

## Pinned primary source

- Official card: <https://huggingface.co/datasets/KLHWONG/vistadata-bimanual>
- Dataset revision: `0884b103c76a3e60ad0dcd9549e56b567a00fedd`
- Card license: CC BY 4.0
- Card scope: contact-rich dual-Franka demonstrations for one
  unscrew-coffee-jar task; GelSight Mini RGB on the right arm's left fingertip,
  left/right wrist RGB, third-person RGB, and robot state.
- Archive: `vistadata-bimanual/unscrew_coffee_jar.zarr.zip`
- Pinned archive size: 8,115,705,227 bytes
- Hugging Face LFS SHA-256: `6b12f4a5c403d58a7fa5c30cdbe0742c33202c24b16cc2cd176e2bb4bd48de18`
- Hugging Face blob OID: `a558587f3aad8ae15de1ad1fa299598fbcbc0ef8`
- Zarr `meta/episode_ends/.zarray` shape: `[30]` (30 episodes)
- No participant/session identity or official split is stated in the card.

Pinned API response receipts are external at
`C:\CodexLocalArtifacts\pretrained-reuse-r2\vistadata-bimanual-metadata-screen-02\`:
`dataset-info.json` SHA-256
`f2e238732f3c17885ad45b46100a5d150b68c4bd474b8c366c547465b0e2d133`, and
`file-tree.json` SHA-256
`883455688011160878b7cfcf514214e9ae7fa88215f2a71b9204640e06a05931`.

## Decision and next gate

This source can support, at most, a small within-task video/tactile pilot after
episode grouping, event-target construction, and stream-necessity controls are
specified without opening any held-out labels. It cannot establish unseen
task, participant, or embodiment generalization on the current public
metadata. Keep the existing fused-readout baselines and negative results
unchanged. Prefer a source with multiple documented independent groups and a
clear task where both inputs are needed before starting another quality fit.


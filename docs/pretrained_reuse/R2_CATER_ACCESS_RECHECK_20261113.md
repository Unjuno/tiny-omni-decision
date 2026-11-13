# R2 CATER access and metadata integrity recheck — 2026-11-13

## Disposition

**Status remains `REVIEW_NOT_ADMITTED`.** This recheck corrects an access-path finding and verifies the publisher's training-subset metadata. It does not authorize video acquisition, fitting, validation scoring, or a quality claim. In particular, the CATER metadata's `CLEVR_new_*` identity namespace overlaps the naming convention of the previously used Clevr-4 source; cross-source scene identity has not been resolved.

No CATER video, Clevr-4 evaluation media, model weight, feature, official CATER validation/test list payload, or sealed audit was opened. No training or GPU work was run.

## Pinned upstream and rights evidence

- CATER repository revision: [`13a19643f1a2fb24e931df25abd74353e4f2fdcb`](https://github.com/rohitgirdhar/CATER/tree/13a19643f1a2fb24e931df25abd74353e4f2fdcb), dated 2020-12-18.
- The [CATER README](https://github.com/rohitgirdhar/CATER/blob/13a19643f1a2fb24e931df25abd74353e4f2fdcb/README.md) states that CATER is Apache-2.0 licensed. The CATER repository includes Blender scene/material/shape assets; several blob hashes exactly match CLEVR assets (`base_scene.blend`, `MyMetal.blend`, `Rubber.blend`, `SmoothCube_v2.blend`, `SmoothCylinder.blend`, and `Sphere.blend`). The [CLEVR dataset page](https://cs.stanford.edu/people/jcjohns/clevr/) states its data is CC BY 4.0. Preserve both attributions.
- The compared CLEVR source tree was pinned to commit `f0ce2c81750bfae09b5bf94d009f42e055f2cb3a`. Matching Git blob IDs include `base_scene.blend` (`e2fa2eab7a0bd14e3ea93c6a0b7ba1439221b31f`), `MyMetal.blend` (`4e6a5b810440c58199e8e7d56bfdb6ab38925dea`), `Rubber.blend` (`b02790c22254e7659d35d0c157a0cad51f023e65`), `SmoothCube_v2.blend` (`b88a02aa3bbd8b6ea1972cf00b90f5005221d044`), `SmoothCylinder.blend` (`9ea5e0d081c3e58546ce80a798e19742b223485c`), and `Sphere.blend` (`4d9189011e4c0bf1248d40af2ec55c681c67218f`). These are exact source-asset matches, not evidence that any CATER video duplicates a Clevr-4 image.
- CATER's data README points to CMU Box static links for `all_actions` videos, scenes, and lists. The initial `cmu.box.com` URL returned 404 in this environment; following its official redirect to `cmu.app.box.com` returned HTTP 200. This corrects the earlier availability statement for this workstation.
- No upstream checksum, ETag, or Last-Modified value was exposed for the video archive. If later admitted, hash every acquired byte and individual selected video locally before feature extraction. The source README's Apache statement is favorable evidence, but no separate terms file accompanies the hosted archive; this remains part of the rights review.

## Accessed files and hashes

The files are stored outside Git at:
`D:\CodexArtifacts\pretrained-reuse-r2\cater-source-screen-20261113-attempt-01\`.

| Official `all_actions` archive | HEAD size | Downloaded | SHA-256 |
|---|---:|---:|---|
| [`lists.zip`](https://cmu.app.box.com/public/static/jr2cc6zomqb01hmzvjztpuip3h0sz8vz.zip) | 116,882 bytes | yes | `564ec049e1bbc548d9ada1c863f58a90bf21f11ed9a861717c08d81a3c5d3495` |
| [`scenes.zip`](https://cmu.app.box.com/public/static/6pfetd8s5k7f6bettrpfv11xiqsgou39.zip) | 163,258,802 bytes | yes | `2855bc5d765516b113dc3bf1cd8c43c7923d3a6350450af816ac6790f4d4f51d` |
| [`videos.zip`](https://cmu.app.box.com/public/static/97kztikj3zwcgv5zu2by6c1jhlvh1d7x.zip) | 14,215,629,625 bytes | no | not downloaded |

The lists archive directory contains official `val.txt` entries. Their names and sizes were visible in the ZIP central directory, but their contents were not extracted or read. From the archive, only `lists/localize_8x8/train_subsetT.txt` and `train_subsetV.txt` were extracted. From `scenes.zip`, only scene JSON entries referenced by those two train subsets were read. No CATER video was fetched.

## Split and scene checks

The upstream `train_subsetT` / `train_subsetV` lists are used as train and selector candidates, respectively; the official CATER validation list remains unopened.

| Partition | Video IDs / rows | Labels | Per-class rows |
|---|---:|---:|---:|
| `train_subsetT` | 3,079 | 64 (`0..63`) | 31–72 |
| `train_subsetV` | 770 | 64 (`0..63`) | 2–21 |

Checks on the selected train subsets:

- Train/selector video-ID overlap: **0**.
- Missing scene JSONs: **0 / 3,849**.
- `image_index` / metadata filename mismatches: **0**.
- Canonical full-scene JSON duplicates within or across subsets: **0**.
- Canonical static object-configuration duplicates within or across subsets: **0**.

Canonical full-scene hashes remove only `image_filename`, `image_index`, and `split`, then hash sorted-key compact JSON. Static-scene hashes retain each object's attributes and initial 3-D coordinates, remove `locations` and `pixel_coords`, sort objects by instance name, and hash canonical JSON. This detects exact normalized metadata duplicates; it does not detect visual near-duplicates or prove independence from other datasets.

Reproducibility artifacts:

- `all-actions-lists.zip`: SHA-256 above.
- `all-actions-scenes.zip`: SHA-256 above.
- `train_subsetT.txt`: SHA-256 `62b1733632a7f447bcc1fcb1a4e1cb4472fabf14d756fe5cabd5df4afddd23cb`.
- `train_subsetV.txt`: SHA-256 `e320cfe0af43d8c1256885096952a4a0bb4e31475332f720a6b530cccd425b81`.
- `scene-integrity.json`: external result SHA-256 `96ae6611db210cf2208a4490b97f4d4b97a86c5e702b7505eaf89708304ecd61`; audit script SHA-256 `86e5700e7d3ed95838d894cc8172c384ff5ab5a73f14ebe5250cee52e5b0f2ea`.

## Remaining admission gate

The selected CATER scene JSONs use `image_filename` values such as `CLEVR_new_000961.avi`; this matches the Clevr-4 image identity namespace already used elsewhere in the project. Filename equality alone does not prove pixel/scene identity, but train/selector IDs cannot be called fresh until the relationship is resolved. Before any video acquisition:

1. Establish from the generator and source manifests whether these IDs are the same underlying scenes as prior Clevr-4 examples.
2. If identity cannot be established from metadata, use only a non-sealed, train-only comparison or exclude any proven overlapping scene groups; do not inspect a sealed audit to resolve it.
3. Reconcile the official CATER archive's Apache-2.0 claim with bundled CLEVR-derived assets and retain the local archive hash if video data is later admitted.

If these checks pass, the next experiment may be one frozen VideoMAE V2 Path A readout on the fixed 8x8 target, with temporal-order, frame-shuffle, position-prior, and option-order controls. It would be synthetic video evidence only and would not satisfy multimodal M-2.

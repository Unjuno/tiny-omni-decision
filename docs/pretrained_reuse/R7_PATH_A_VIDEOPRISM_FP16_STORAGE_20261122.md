# R7 VideoPrism FP16 storage diagnostic — 2026-11-22

## Outcome

An isolated image bundle stores VideoPrism's 114,365,184 weights in IEEE
float16, then casts the loaded values back to float32 for inference with the
official pinned JAX/Flax runtime. The checkpoint is 228,729,985 bytes smaller
than the original image bundle payload. Full replay on the already-exposed
512-image CLEVR-4 checkpoint-selection validation found one changed decision
out of 2,048. This is a precision diagnostic on exposed synthetic development
data, not independent quality evidence or a product release.

The strict runtime parity gate **failed**: maximum probability delta was
`0.0121371`, above the unchanged `1e-5` threshold, and one prediction changed.
Accuracy moved from `87.8906%` to `87.9395%` because that changed prediction
became correct. The small observed metric movement does not waive the parity
failure.

## Inputs and integrity

- Source assembly: `path-a-modular-runtime-assembly-v4-whisper-encoder-only-20261122-attempt-01`.
- Source package manifest SHA-256:
  `616b9b61e983d875a6a29e71d568f1ba0a6aeba6705da61ee66a88b4be43b3fd`.
- Source image bundle manifest SHA-256:
  `958e1f8735c33237342d0533f7529a0514fc4f002ca812057b2eff28ac51012e`.
- Source VideoPrism checkpoint SHA-256:
  `61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29`.
- Source model revision: `google/videoprism-base-f16r288` at
  `be719a406d563b66f0ac969e7c94bab8e997c81a`.
- VideoPrism source checkout: Apache-2.0 at
  `c6eba0ece3dd49a24ffc44582966997a4a59f1d8`.
- The existing task is static CLEVR-4; each image is repeated over eight input
  frame slots. It is not natural-image or temporal-video quality evidence.
- The exact baseline validation predictions SHA-256 is
  `2336f6efafe4b6d5e1efa4f1257b5f9eb39c03824d7c4a53fda31a920162d30a`.
  The candidate was compared on those same 512 checkpoint-selection IDs and
  order. The set is already exposed and checkpoint selection used it.

## Export and runtime behavior

The new builder refuses to overwrite outputs, checks the source manifest and
checkpoint hashes, requires all 40 arrays to be finite float32, casts to
float16, and verifies the resulting bundle file list and hashes. The bundled
runtime casts restored weights back to float32 before the same JAX/Flax model
forward. This isolates storage precision rounding from inference arithmetic;
it does not reduce resident float32 inference weights.

The verifier previously imported the checkout's runtime because the source
package was already imported, instead of loading the runtime in the bundle.
It now loads the bundle module directly without writing bytecode into the
immutable bundle. The small regression test covers this. An initial local
builder constant also transcribed one checkpoint hash character incorrectly;
the fail-closed check stopped before creating output, and the constant was
corrected against both source manifests and the actual file. The earlier
storage-dtype audit now records the verified hash.

## Paired development metrics

The table reports the full 2,048-decision selection-validation slice. It is
not a fresh benchmark; no checkpoint selection, calibration, or training was
performed.

| Metric | float32 baseline | float16 storage / float32 compute |
|---|---:|---:|
| Accuracy | 0.878906 | 0.879395 |
| NLL | 0.287809 | 0.287782 |
| Multiclass Brier | 0.154222 | 0.154213 |
| ECE-15, equal width | 0.021868 | 0.022162 |
| Mean confidence | 0.869332 | 0.869326 |
| Changed argmax decisions | — | 1 / 2,048 |
| Maximum probability delta | — | 0.012137 |

| CLEVR-4 taxonomy | Accuracy FP32 storage → FP16 storage | NLL FP32 → FP16 | Brier FP32 → FP16 | ECE-15 FP32 → FP16 |
|---|---:|---:|---:|---:|
| Color | 0.994141 → 0.994141 | 0.027247 → 0.027261 | 0.010571 → 0.010583 | 0.021121 → 0.021112 |
| Count | 0.607422 → 0.609375 | 0.880738 → 0.880779 | 0.486667 → 0.486678 | 0.061589 → 0.061467 |
| Shape | 0.996094 → 0.996094 | 0.013576 → 0.013571 | 0.005459 → 0.005456 | 0.008231 → 0.008230 |
| Texture | 0.917969 → 0.917969 | 0.229675 → 0.229518 | 0.114189 → 0.114136 | 0.023030 → 0.021697 |

The count taxonomy remains the dominant weakness, at about 61% accuracy. The
model is far below the aspirational 95–98% per-modality goal even on this narrow
synthetic image task.

## Size and runtime cost

| Measure | Existing four-route assembly | Hypothetical assembly with this FP16 image bundle |
|---|---:|---:|
| Payload bytes, excluding outer manifest | 594,168,210 | 365,438,780 |
| Provisional 200,000,000-byte target | Not met | Not met |
| Unique logical runtime parameters | 163,102,597 | 163,102,597 |

The hypothetical four-route payload replaces only the image bundle's
checkpoint, runtime, and nested manifest sizes with the verified values from
the new image bundle; no full four-route variant was built. It remains
165,438,780 bytes above the size target. The isolated image bundle itself is
229,413,804 bytes, also above 200,000,000 bytes.

The 512-image replay took `791.16` CPU seconds (`1.545` seconds/image). Peak
working set was `4,274,933,760` bytes. No GPU, cloud compute, training, or new
media was used. FP16 storage did not reduce runtime memory because weights are
expanded to float32 after loading.

## Artifacts and verification

- Canonical isolated FP16-storage bundle:
  `C:\CodexArtifacts\pretrained-reuse-r2\path-a-image-videoprism-fp16-storage-20261122-attempt-02\`
- Its checkpoint SHA-256 is
  `8b38cdf070a0d9c9ffb3254e91da510e26f8176290d82c0572914f96ee9bae42`;
  manifest SHA-256 is
  `ec227c59e9fa849c6113e950cadd0e4e854f228da99156159d0bf206998b6e33`.
- Full replay used the bit-identical checkpoint/runtime from bundle attempt 01
  (manifest SHA-256 `d272ae21e34c55e82df6745f4b84175396ff13d312f3f428892ef096c4891bbb`);
  attempt 02 differs only in manifest creation metadata.
- Full replay result SHA-256:
  `83e19b1937fb77d294156ebd6d92d75c9f6124aa3a550866662d9646e2632075`.
- Prediction JSONL SHA-256:
  `9fefe37cdf3636ed46ebec283b993151cc1be419d30c2ff546ed0950dc8da749`.
- Paired metrics JSON SHA-256:
  `73462b737d857127d1f5476735207a15a34c0f14088d57bcff31e48594d1558b`.
- No validation media was copied or modified. The baseline package, weights,
  checkpoints, train data, and sealed audit were left unchanged.
- Focused tests: 8 passed. Ruff passed on the changed script and tests.
- Implementation source commit before this experiment: `e5e760d49aaac38ed13499cbb4b197b077e45eae`.
- Builder SHA-256: `f77e1bb768c28501b5eda6e42a0cda92ae200d3165bc25f153c33fcd7d9a4496`.
- Verifier SHA-256: `a1f1d7977c6fcbdc101d0269d632cdb74bd6e864c6f6eb28d988495ca6f57443`.
- Metrics summarizer SHA-256: `dd4c4137a84134756bc593a25a6c804424e09a9837561a5674168d5f87938ac1`.

## Decision

Keep the established float32 bundle as the runtime reference. The FP16 export
is a useful storage proof of concept but fails the existing strict parity gate
and leaves the assembled package well above target. Do not promote it or call
the product size goal achieved. Further quantization would need its own
separate artifact, unchanged thresholds, paired task metrics, and a complete
four-route payload measurement; the current image quality ceiling and the
other modality/task gates remain open.

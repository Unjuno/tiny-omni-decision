# X-Capture frozen feature runtime smoke — 2026-11-28

## Result

Processed the previously frozen 20-object train/validation sample on CPU using
the pinned TinyCLIP visual tower and EfficientAT DyMN-0.4 audio encoder. This
only verifies that the selected source data passes the real frozen encoders and
that finite, serializable features can be produced. It is not training, model
selection, retrieval evaluation, or evidence of task quality.

| Input | Encoder | Feature shape | Forward time, 20 samples |
|---|---|---:|---:|
| Masked RGB object crops | TinyCLIP ViT-39M vision tower | 20 × 512 | 1.574 s |
| 10N tactile images | Same TinyCLIP visual tower | 20 × 512 | 1.491 s |
| Normalized impact audio, 48 kHz → 32 kHz | EfficientAT DyMN-0.4 | 20 × 384 | 0.980 s |

Total measured process time was 4.892 seconds on CPU, including encoder load,
preprocessing, I/O, array writes, and reload checks. GPU use and training were
both false; cloud cost was $0. The full CLIP checkpoint (83,101,185 parameters)
was loaded locally, while only its visual tower produced image/tactile features.
The audio checkpoint has 1,961,239 parameters. The latter tactile route is an
engineering reuse probe, not tactile-specific pretraining.

Every output vector was checked for finite values. All three NumPy arrays were
reloaded and compared element-for-element with their in-memory values. The
sample contains 10 train and 10 validation points from 20 distinct object IDs;
no evaluation ID or media was opened.

## Pinned inputs and outputs

- Source commit and output-attempt receipt are recorded in the final run
  section below after a fresh execution of the committed script.
- X-Capture revision: `b4468bf999aac250f34ef1b02459d902492a068f`
- Sample receipt SHA-256: `17bacf80527a1aa0e1496eb330637d7f81b039ca68ffa28ccfabf9b33b240c0f`
- Masked-crop receipt SHA-256: `6c37406b3e8011bab5fd564470b5b8d312d4016966906a1f2683f723d4453c84`
- TinyCLIP revision: `07a4b0bc751cb64fecd2b661c048c1dd98d69444`
- TinyCLIP weights SHA-256: `112bdfb07ebc8994e9afba7a2a314e278493a675033595cf48e54cf8a17ae587`
- EfficientAT source revision: `a425fdce92572e602a1d5634799bd9f1f2efa806`
- EfficientAT weights SHA-256: `49b0e63577021994257b3de2ac5f1686d71554b93c8149f9d2568ad5760b904e`
- Runtime: Python 3.11.9, PyTorch/torchaudio 2.6.0+cu124, Transformers 5.6.2;
  execution device CPU.
- Output directory:
  `C:\CodexArtifacts\pretrained-reuse-r2-local\xcapture-frozen-feature-smoke-20261128-attempt-04`
- `image-features.npy` SHA-256:
  `b04fd158d8c2bc75e9278e953bc77d006aaa3fca69d88b6e75408c40f7618a21`
- `audio-features.npy` SHA-256:
  `10716382fff49f2a667d5fabbf9a40fc3ab9d82aaebea53a2995fa7bf4defff1`
- `tactile-features.npy` SHA-256:
  `a3c037f68395b281cd8f79050219b91903808e4f08fd4bb18c46a30230e9b388`
- Feature receipt `feature-smoke.json` SHA-256: recorded after the committed
  script's final execution below.

## Data and implementation notes

The existing visual audit found readable package/brand text in at least 5 of
20 manually viewed masked crops. This smoke deliberately uses those same
object-only masked crops and does not claim that their contents are shortcut
free. Before a retrieval fit, the primary comparison must include a frozen
text/mark-occlusion control and cross-point positives so the model cannot
benefit merely from pairing the same interaction timestamp. The sample is far
too small for fitting or quality claims; the next data step must use a
predeclared, object-disjoint train/validation selection with enough distinct
objects and points for grouped evaluation.

Windows' installed torchaudio backend could not directly decode the dataset's
IEEE-float WAV, and no codec package was installed. The smoke therefore uses a
small, strict RIFF IEEE-float32 decoder and the already installed torchaudio
resampler. Three focused CPU tests cover correct decoding and fail-closed
behavior for unsupported/truncated WAV files. No dependency was installed or
upgraded.

## Reproduction

From the R2 X-Capture worktree, with the pinned R1 dependency directories on
`PYTHONPATH`:

```powershell
$env:PYTHONPATH='C:\CodexArtifacts\pretrained-reuse-r1\python-deps;C:\CodexArtifacts\pretrained-reuse-r1\efficientat-audit-v1\source;.'
python scripts\extract_xcapture_frozen_feature_smoke.py
```

The script refuses to overwrite an existing output and verifies the pinned
manifests, source revisions, model hashes, each media/crop hash, train/validation
partition membership, and unique object IDs before encoding.

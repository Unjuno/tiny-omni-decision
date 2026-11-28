# Physion++ matched-pair low-level cue screen — 2026-11-28

## Scope and protocol

This is a train-only data-quality screen over the existing fixed set of 75
exact-recorded-factor matched No/Yes video pairs (150 distinct clips). It
re-read only the existing 652-row train manifest and the 75-pair selection
manifest, verified each selected file against its declared SHA-256, and decoded
eight uniformly spaced frames from each video. It did not read validation,
publisher test, or sealed data; it did not load a model or train a readout.

The predeclared feature family contains ten low-level cues: compressed file
bytes, decoded frame count, duration, average FPS, sampled-frame mean R/G/B,
BT.709 luma standard deviation, near-black/near-white pixel fraction, and mean
absolute luma difference between adjacent sampled frames. Frames were reduced
to 32×32 with aspect ratio preserved. For every feature, the comparison is the
paired mean difference (Yes minus No), with a two-sided sign-flip permutation
test (20,000 permutations) and paired-pair bootstrap percentile interval
(20,000 resamples). Holm correction covers all ten features. Seed: `20261128`.

## Results

| Feature | Mean paired Yes−No | 95% pair-bootstrap interval | Standardized paired effect | Holm p |
|---|---:|---:|---:|---:|
| File bytes | +2,798 B | −3,054 to +8,914 B | 0.105 | 1.000 |
| Decoded frame count | −2.91 frames | −12.64 to +7.21 | −0.066 | 1.000 |
| Duration | −0.097 s | −0.421 to +0.240 | −0.066 | 1.000 |
| Average FPS | 0 | 0 to 0 | 0 | 1.000 |
| Mean red | −0.00007 | −0.00763 to +0.00758 | −0.002 | 1.000 |
| Mean green | −0.00392 | −0.01052 to +0.00278 | −0.133 | 1.000 |
| Mean blue | −0.00218 | −0.00741 to +0.00324 | −0.092 | 1.000 |
| Luma standard deviation | −0.00300 | −0.00677 to +0.00083 | −0.174 | 1.000 |
| Clipped-pixel fraction | +0.00151 | −0.00273 to +0.00571 | 0.082 | 1.000 |
| Adjacent sampled-frame luma delta | −0.00219 | −0.00540 to +0.00106 | −0.152 | 1.000 |

No feature showed a detectable paired difference after the predeclared
multiple-testing correction. The strongest absolute standardized effects were
small (at most 0.174 in this feature family); every unadjusted interval shown
includes zero.

## Interpretation and limits

This reduces concern about a large, simple label-correlated cue in these ten
coarse statistics within the selected exact-recorded-factor pairs. It does not
show that low-level shortcuts are absent, that all unrecorded simulator or
rendering factors are balanced, or that the target is reliably observable.
There are only 75 matched pairs; 32×32 summaries of eight frames can miss local
contact, geometry, motion, texture, codec, and higher-order signals. The
selection is from previously observed train data, so this is neither an
independent quality result nor a blind audit.

The publisher label/timeline alignment and train/validation identity checks
remain valid for their stated scopes. The independent human blinded review is
still incomplete; until it is reviewed, the data-observability gate remains
open. In particular, the prior uniform 8-frame contact-event coverage result
(63.6% exact positive contact-frame inclusion) is not resolved by this
low-level cue screen. Do not use this result to justify a BERT sweep, declare a
video-model ceiling, or claim general multimodal quality.

## Reproduction and provenance

```powershell
python scripts/audit_physionpp_low_level_pair_cues.py `
  --train C:\CodexArtifacts\tqpp\fullclip-v1\train.jsonl `
  --pairs C:\CodexArtifacts\pretrained-reuse-r2-local\physionpp-train-data-audit-20261128-attempt-06\visual-review-pairs.jsonl `
  --corpus-root C:\CodexArtifacts\tqpp\fullclip-v1 `
  --output C:\CodexArtifacts\pretrained-reuse-r2-local\physionpp-low-level-pair-cues-20261128-attempt-01\audit-report.json
```

- Train JSONL SHA-256: `0de1bbbb1cc54000936c1771ea54e375c214fedda36734536f31a44c4f64932b`
- Pair manifest SHA-256: `38766b31a0b0af9048479236d556724ca47c374f9cedbb905780cebfe2be1333`
- Script SHA-256: `76eae6e5e3d1ede387e9948ee6c5240b3f97e1646dca65a9eb894c40e2bb1019`
- Focused tests: `tests/test_audit_physionpp_low_level_pair_cues.py` — 6 passed
- Physion++ audit regression group: 60 passed; Ruff and `git diff --check` passed
- Machine report: `C:\CodexArtifacts\pretrained-reuse-r2-local\physionpp-low-level-pair-cues-20261128-attempt-01\audit-report.json`
- Machine report SHA-256: `a6ce33a36693a9c40465a8a6c836ebe6ddd777e618160c35ed8bd88f1716463e`
- Runtime: system Python 3.11.9, PyAV 18.1.0, NumPy 2.4.6, Pillow 10.4.0; CPU only
- No model/GPU execution or process intervention occurred. Existing Orbitune and
  the unrelated IntPhys download process were left untouched.

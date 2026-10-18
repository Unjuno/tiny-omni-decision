# Full reference waveform screen: frozen protocol, not yet executed

Prepared `scripts.check_fsd_starss_waveforms` to extend the verified waveform
primitive from eight selected neighbors to every fixed FSD clip against every
recording in the unchanged STARSS development manifests. Requires complete128
media acquisition with exact selected-ID order and source SHA256 checks. Partial
progress files cannot satisfy the entry gate. All reference recording hashes
are verified before decoding. No model probabilities or labels affect matching.

- Native FSD mono and STARSS channel0, torchaudio resampled to2kHz.
- Entire FSD clip slid across each complete STARSS recording.
- Absolute normalized correlation, invariant to gain, constant offset and polarity.
- Flag correlations >=.99 or undefined silent-signal comparisons for review.
- Stop on missing/corrupt media or template longer than reference; no skipped pairs.
- Also record exact duplicates among resampled FSD clips.
- Save recording-level progress and terminal results in a new directory.

The threshold was already fixed for the smoke. No threshold fitting or example
removal is planned. A completed screen without flags would mean no matching
whole-clip waveforms under this transform, **not** a proof against edited,
time-stretched, mixed or shorter reused excerpts. Source provenance and these
limitations must accompany any subsequent quality result.

```powershell
$env:PYTHONPATH='C:/CodexArtifacts/pretrained-reuse-r1/python-deps'
python -m scripts.check_fsd_starss_waveforms
```

Prepared output: `C:/CodexArtifacts/pretrained-reuse-r2/fsd-starss-waveform-screen-128-01/`.
This script has not run because acquisition is still live. No learning started.
Ruff passes; current CPU suite755passed/1skipped, including the correlation
primitive tests. That suite does not establish this full-data screen's outcome.

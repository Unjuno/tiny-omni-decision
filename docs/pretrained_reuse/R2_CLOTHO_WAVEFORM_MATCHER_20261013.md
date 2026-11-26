# Measured FFT reuse for the pending full waveform screen

Exhaustive full-corpus matching requires180,004 cross-split pairs. Each original
comparison recomputed both waveforms' Fourier transforms. Added a prepared
matcher that caches the same raw and centered/reversed transforms, energy and
prefix statistics with a common sufficient FFT length. Valid-offset search,
gain/DC/polarity normalization and .99 review threshold are unchanged.

Executed120 pairs from the16 verified training probe clips at mono2kHz:

| CPU execution | Seconds |
| --- | ---: |
| Original pairwise implementation | .734724 |
| Preparation | .058945 |
| Prepared comparisons | .228592 |
| Prepared total | .287537 |

Measured speedup2.555× including preparation; this is a small-corpus measurement,
not guaranteed full-run throughput. Cache45,142,432bytes. All offsets and .99
review decisions identical; max correlation difference6.94e-18. Four targeted
tests also cover unequal/equal lengths, planted gain/polarity/DC at the last
valid offset, constant signals, reversed lengths, invalid inputs and memory cap.
Ruff passed. No model training or label metrics were used for this optimization.

The full scanner uses a3GiB cache cap (plus waveform data and worker scratch),
records actual cache bytes, and fails before allocation if the cap is exceeded.
The helper default cap remains2GiB. Existing full media acquisition is not
modified; the full waveform scanner has not started. No guarantee is made
against edited/mixed/time-stretched or shorter shared excerpts.

Artifact: `C:/CodexArtifacts/pretrained-reuse-r2/clotho-waveform-matcher-benchmark-01/`.
Exact executed benchmark script is `clotho_waveform_matcher_benchmark.py` in
the parent directory. Initial invocation failed before creating output because
the repository import path was absent; adding process-local PYTHONPATH resolved it.
Compact measured report accompanies this document.

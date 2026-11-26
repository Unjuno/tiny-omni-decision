# Fixed spatial audio direction diagnostic

Executed the algorithm frozen at **9f3d3d9** on all49 pinned STARSS23 training
recordings. No model fitting, filter search, room-dependent transform or seed
search. No calibration/validation/audit data accessed. This is a diagnostic
for a possible spatial feature path in the main Omni model, not a separate
audio-model project or a completed audiovisual task.

## Method and scope

Read verified4-channel24kHz PCM16 FOA data in its documented W,Y,Z,X order.
For each100ms block, compute mean products W*[X,Y,Z] and normalize the resulting
three-vector. No filtering, frequency weighting or denoising. Compare spherical
angular error with the one annotation at that tick. Multiple-annotation ticks
are explicitly outside this single-direction diagnostic, not removed from any
existing benchmark. A single annotated target does not guarantee a single
physical source: interference, reverberation and non-target sound remain.

Unit tests confirm known ideal +X/+Y direction responses, amplitude-scale
invariance, zero-direction handling and rejection of malformed/nonfinite data.
These establish implementation convention, not real room calibration.

## Measurements

| Quantity | Result |
|---|---:|
| Complete audio ticks | 99,082 |
| Annotated ticks | 81,810 |
| Multiple-annotation ticks excluded from this diagnostic | 27,058 |
| Zero direction ticks | 12 |
| Incomplete audio ticks | 2 |
| Scored single-annotation ticks | 54,738 |
| Mean angular error | 61.0036 degrees |
| Median angular error | 52.7515 degrees |
| Within20 degrees | 20.51% |
| Within45 degrees | 45.20% |

Per-room median errors39.88..63.89degrees. No high-precision spatial adoption
is justified. This result is not a ceiling for spatial audio models or proof
that an individual preprocessing factor caused the error. The many adjacent
ticks are correlated and must not be counted as54,738 independent trials.

A separately recorded **post-hoc** sign sanity check uses saved predictions
only: negating the full predicted vector raises mean error to118.9964degrees
(median127.2485). This rejects the idea that a simple global antipodal flip
alone fixes these predictions; it does not validate every axis, clock, room
alignment, source assumption or generalization. No chosen sign was fitted from
the outcome; the original documented convention is unchanged.

## Artifacts and reproduction

```text
python -m scripts.probe_starss23_foa_direction
```

`C:/CodexArtifacts/pretrained-reuse-r2/starss23-foa-direction-probe-01/` contains
config/pins, one prediction record per scored tick, aggregate/per-room metrics,
and separate posthoc-antipodal-check.json. The runner verifies training split,
training window manifest, annotation archive and every audio file hash. The
previous training metadata audit verified the complete six-column schema.
Predictions SHA256:
`c6ae866a84f1c6dc80eb78044cd660d402b59583f9eb780785758875109c0841`.

CPU execution5.286seconds for verified media loading, feature extraction,
comparison and saving (initial metadata pin checks excluded). No GPU, learned
parameters or cloud fees. The physical feature adds no model parameters, but
has not established a useful Decision quality gain. It would only be available
after its100ms evidence block and calculation finish.

## Consequence for next work

Do not inject this estimate as trusted spatial truth, generate audiovisual
labels from it, or claim joint reasoning from the mono event scores. A next
spatial experiment needs an independently frozen method with time/frequency
structure and explicit polyphony handling, then a grouped development check.
Temporal alignment and visual source association remain separate unresolved
requirements. Existing event readouts, video caches and quality metrics remain
unchanged; all-modality95–98% is still unachieved.

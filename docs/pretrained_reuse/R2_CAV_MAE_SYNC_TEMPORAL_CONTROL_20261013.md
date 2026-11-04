# Saved-feature temporal correspondence control

Post-hoc development diagnosis, no training or new media extraction. Reverse
the order of eight audio feature windows while retaining video order, candidates,
weights and temperature1. This is feature-window order reversal, not reversing
waveforms or motion within a frame/window.

| Metric | Matched | Audio window order reversed |
| --- | ---: | ---: |
| Accuracy | .3500 | .3250 |
| NLL | 1.371106 | 1.371022 |
| Brier | .742384 | .742336 |
| ECE | .086390 | .061371 |

Reversed-minus-matched paired quartet bootstrap (50 groups,10000 draws,seed17,
unadjusted percentile95%): Accuracy−.025, interval[−.065,.015]; NLL−.0000833,
interval[−.0009997,.0008101]. Matched replay probabilities are bitwise identical
to the previous saved predictions. Input features and routes are hash-checked.

No clear useful temporal correspondence is established in this configuration.
Static/slow events, overlapping audio windows and a narrow development task
limit interpretation. This does not prove the pretrained model lacks temporal
information. Close this readout comparison without parameter/temperature search
or product promotion. Full upstream16-frame evaluation is untested, and semantic
joint decision quality remains unmet.

Command: `python -m scripts.audit_cav_sync_temporal_control`.
Artifacts: `C:/CodexArtifacts/pretrained-reuse-r1/cav-mae-sync-temporal-control-01/`.
Both conditions' predictions, metrics and bootstrap settings are retained.
No existing artifact, model weight, or evaluation split changed.

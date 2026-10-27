# Same-label audio substitution on train-only OOF checkpoints

The [fixed protocol](R2_SOURCE_SAME_LABEL_AUDIO_PROTOCOL_20261027.md) reused the
three saved joint cross-validation checkpoints. For each eligible held-out
record, a source-annotation label was joined from the pinned official-train
archive and a donor audio feature with the **same label** was selected from a
different video in the same fold. Labels selected donors only and were not fed
to the model. Original predictions were verified against saved scores before
evaluating substitutions. No learning, checkpoint selection, official
validation, or sealed audit occurred.

Of 2,593 records, **2,581** could be matched: 11 had missing or mixed source
labels and one had no other same-label video in its fold. All 12 remain
unmatched; no claim is made about them. The matched subset spans 320 video
groups and 15 label IDs; some labels have very few examples. Source labels can
be broad and same-label recordings need not share physical source or context.

| Input on the same 2,581 matched records | Accuracy | Visible-source accuracy | None-visible accuracy |
|---|---:|---:|---:|
| Original audio | 0.5498 | 0.5063 (1,900 records) | 0.6711 (681 records) |
| Same-label audio from another video | 0.5525 | 0.5011 | 0.6960 |

Original minus substituted Accuracy was −0.27 percentage points, with a 95%
paired video-group percentile bootstrap interval of −1.79 to +1.27 points.
Set-NLL difference was +0.00084, interval −0.0450 to +0.0451. The diagnostic
therefore found **no clear aggregate performance loss** under this same-label
substitution, within the uncertainty and coverage of this sample. It does not
prove the model ignores audio or prove equivalence. The previous unrestricted
different-video replacement lowered Accuracy, but could change the sound label;
these results suggest class-level cues or other differences may explain some of
that earlier sensitivity. Treat this as a hypothesis, not causal proof.

This is nonblind internal cross-validation, one seed, with overlapping fold
training sets, oracle visual candidates, possible common participants and
unverified source-label fidelity. Group bootstrap used 10,000 draws, seed 17;
it does not account for training variability or all source dependencies.
Independent validated both-input task evidence is still needed, and the
95–98% four-modality objective remains unmet.

The CPU-only result is
`C:/CodexArtifacts/pretrained-reuse-r2/perception-source-same-label-interventions-20261027-attempt-01/result.json`,
SHA-256 `30471d27f3c9dff4ba6427dbc5fe734bda97927a5b55ec554733252e64ed8b24`.
It records per-label/fold coverage, fixed donor-map and prediction hashes,
protocol and code hashes, and paired intervals; no prior artifacts changed.
Ruff and four targeted tests passed. CUDA was not initialized.

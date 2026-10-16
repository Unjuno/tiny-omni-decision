# Greatest Hits: complete TRAIN label inventory and development media acquisition

## Completed work

The prior three-parent probe was extended to **all733 official TRAIN annotation
members**, using exact version-checked ZIP ranges. All733 parsed successfully;
730 new members were acquired and3 existing verified members reused. Transfer:
348,254 bytes; wall time133.282 s. No official TEST media/labels were opened.
Source:`f4130e200ef390bb38a3bf16667e465896751329`.

The date-grouped development partition was committed before the remaining labels
were read. Preserve the324 TRAIN parents whose timestamp prefix also appears in
official TEST as quarantine. Of the remaining409, the fixed five dates produce
101 development-validation parents;308 parents remain in training. No label-based
rebalance or seed search occurred. IDs and hashes are saved.

| Partition | Parent videos | Annotation events | All3 fields known | All3 fields missing |
|---|---:|---:|---:|---:|
| TRAIN | 308 | 13,327 | 7,551 | 5,225 |
| Development validation | 101 | 4,605 | 2,800 | 1,556 |
| Quarantine | 324 | 16,789 | 9,906 | 6,011 |
| All official TRAIN | **733** | **34,721** | **20,257** | **12,792** |

The remaining1,672 events have partial labels; missing values remain missing.
Every source row is retained in the inventory. Date/timestamp-prefix and parent
counts are available in the local full inventory, including quarantine.

## Fixed native task support

| Task | Options retained | Known TRAIN fields | Known validation fields |
|---|---:|---:|---:|
| Material | 17 | 7,928 | 2,978 |
| Action | 2 | 7,946 | 2,987 |
| Reaction | 5 | 7,833 | 2,910 |
| Total prospective questions | — | **23,707** | **8,875** |

Every validation label value has TRAIN support. Material carpet and tile have
TRAIN support but no positive validation examples; this validation cannot measure
their recall. All17 options remain present. Native questions ask for the material,
action or reaction at a supplied contact event. They do not include the target
label in the question; target values are separate supervision fields.

The prospective corpus builder preserves every known field, including partially
annotated events, and fails on invalid windows or unknown classes instead of
silently reducing the corpus. **These counts describe annotation support, not a
completed corpus build or a trained model.** A separate label-prior diagnostic
uses TRAIN frequencies only; it has not yet been run on a completed task corpus.

## Measured distribution shift

TRAIN and validation have substantially different distributions. For example,
water has109 TRAIN labels and486 validation labels; dirt58 and310. Static reaction
has5,775 TRAIN labels and1,229 validation labels. Do not change dates, class counts
or options to make the evaluation easier.

The plot of known-label proportions is saved at
`C:/CodexArtifacts/pretrained-reuse-r2/greatest-hits-development-label-plot-01/label-distribution.png`.
It uses the complete fixed inventory, not model predictions. Its initial invocation
could not import matplotlib from the default dependency path; the existing pinned
plot dependency directory was then used successfully. No installation or upgrade.

## Actual media acquisition completed

All409 development parents' `_denoised.mp4` files were acquired. Extracted size
**643,096,543 bytes**; actual range transfer632,309,869 bytes. All exact206/range/
ETag/size/CRC checks passed and every file has SHA256 evidence. Largest extracted
member32,950,656 bytes fits32MiB. Two workers took**568.687 s**.
Source:`8ce7de7e8ae67f44d525729261af4a2a46827dd4`.

The explicitly registered media request cap is90s (subprocess95s), while the prior
annotation default remains25s (subprocess30s), with regression coverage. TLS was
enabled and IPv4 was used. No full archive or quarantined/official TEST media was
downloaded. This is a partial archive acquisition; there is no full-archive hash.

## Full integrity decode is a separate running stage

The all409-parent CPU audit was launched with source
`7de98c5` and output
`C:/CodexArtifacts/pretrained-reuse-r2/greatest-hits-development-decode-01`.
This publication **does not claim that audit has completed**. Its eventual
`result.json`, per-parent receipts and live process status are authoritative.
The active call's process started2026-10-16T17:04:33UTC; execution-session3404,
launcherPID15636 / Python workerPID3300. Verify identity/liveness before using
those handles; never restart because a status observation timed out.

The audit reads all decoded frames and PCM samples, records finite/monotonic
streams and content hashes, rejects exact decoded content overlap across splits,
and accounts for every prospective0.5s window `[onset−0.25,onset+0.25)` in both
streams. Published annotations have minimum adjacent gaps>0.25s, so another
annotated onset is outside the window interior. Unannotated sounds and sustained
scratches can still occur; source onsets are supplied localization, not a learned
event detector. The corpus builder requires the completed audit and no invalid
windows. No model or encoder has been loaded for these stages.

## Limits and next action

License evidence remains the [official Greatest Hits author release](https://andrewowens.com/vis/).
The [primary paper](https://arxiv.org/abs/1512.08512) describes impact-centered
0.5s prediction and semantic annotations. This motivates a bounded task definition;
it does not establish our model's accuracy, exact physical synchronization or
the validity of every semantic label.

Exact physical AV offset and repeated sites/actors across dates remain UNKNOWN.
Date grouping is conservative and reproducible, not proof of independent scenes.
Material/action/reaction tasks are not assumed to require both modalities; that
must be measured against audio-only, video-only and question-only controls.
Keep the gate comparison's stricter data/intervention criteria open. No neural
training, sealed audit, paid compute, product promotion or95–98% claim.

Next: observe the existing decode through terminal completion; inspect every
failure/window/content result. Only if the fixed integrity conditions pass, execute
the already-tested task builder in a new output directory. Preserve the original
Attention baseline, conditional gate arms, all existing checkpoints and validation.
Then verify actual event preprocessing/features and define valid intervention
labels before registering any matched gate learning curves.

## Evidence and verification

[Small inventory/acquisition manifest](../../manifests/pretrained_reuse/greatest-hits-inventory-20261017.json)
contains split/label counts, exact commands, revisions, source hashes and local
receipt hashes. Full receipts and media stay outside Git under
`C:/CodexArtifacts/pretrained-reuse-r2/greatest-hits-*`.

- Partition file SHA256:`6c9ac8eb8e12bd5dc0af77c8d162a40c753d5a215d07eb85dc3a69c3973d9f98`.
- Annotation JSON SHA256:`24e8df777bee34102591e7b15901b931034ee2df3b549d632fb5b7094322e3dd`.
- Native model/trainer/corpus/checkpoints were not changed.994 protected prior
  artifact hashes were rechecked without changes; original user worktree preserved.
- Local CPU suite:2,225 passed/1 skipped; all25 existing manifest commands passed.
  The new inventory JSON is an evidence snapshot, separate from the legacy CLI
  dataset-manifest schema. Ruff and exact-push CI results are saved in the
  `greatest-hits-development-verification-01` publication receipts.

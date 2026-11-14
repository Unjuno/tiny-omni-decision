# TartanAviation ATC + ADS-B metadata feasibility — 2026-11-14

## Disposition

Keep this source/task **`REVIEW_NOT_ADMITTED`**. The official object index
shows potentially synchronized airport-day availability, but does not establish
1,000 usable examples, 200 independent groups after aircraft/speaker grouping,
reliable labels, or that a model must use both inputs. A prior local metadata
screen of the third-party label derivative has exposed all 184 pinned Parquet
shards and aggregate clip/date identities; this is now an additional source
exposure that must be reconciled before any future reuse. It did not read
transcript or callsign text, or media. No TartanAviation media or trajectory
rows were downloaded, opened, or decoded for this audit; no fitting or model
evaluation was run.

## Pinned source and rights

- Official source repository: `castacks/TartanAviation`, main revision
  `077a04abac1e50953c45b72524edd79e3a21d9bd` (the current official main at
  audit time).
- The pinned README states that the dataset, including speech and ADS-B, is
  CC BY 4.0. The repository code has a separate BSD-3-Clause license; the code
  license is not the data-rights basis.
- Official source describes daily audio archives and daily ADS-B directories
  inside annual ZIPs. It documents audio clip start/end/duration metadata and
  timestamped trajectories, but no supplied transcripts or audio-to-clearance
  labels. [Official repository](https://github.com/castacks/TartanAviation/tree/077a04abac1e50953c45b72524edd79e3a21d9bd),
  [official AirLab data description](https://theairlab.org/tartanaviation/).

## Read-only object-index result

The audit used the public Swift JSON object listing and strict HTTP Range
requests for ZIP central-directory tails. It requested five 8 MiB suffixes
(41,943,040 bytes total) to read archive member names only. The reader refused
any response that was not the exact requested `206 Partial Content`. It did
not download/extract ZIPs or read WAV/CSV members.

| Source index | Airport/date groups |
| --- | ---: |
| Audio daily archives, KAGC | 399 |
| Audio daily archives, KBTP | 278 |
| ADS-B dates, KAGC | 279 |
| ADS-B dates, KBTP | 381 |
| Audio and ADS-B date intersection, KAGC | 229 |
| Audio and ADS-B date intersection, KBTP | 268 |
| Total intersection | 497 |

The 497 airport-date intersections exceed the provisional 200-group count only
as an **availability upper bound**. They are not yet independent groups:
recurrent aircraft and radio speakers may connect dates across partitions.
No example count for a usable spoken instruction/response pair or human-verified
task label can be inferred from the object index.

## Task hypothesis and failure controls

One candidate task is: **given an ATC audio instruction and anonymized
candidate ADS-B trajectories, select which candidate trajectory followed the
spoken maneuver**. Audio would specify the intended instruction; trajectories
would show which candidate acted on it. The options should use randomly
permuted, per-example opaque IDs. A temporary callsign-to-opaque-track mapping
may be required as sensor input so that the audio reference can be grounded;
stable aircraft IDs and tail numbers must not persist across samples or appear
as option labels. A label must be independently human-checked against audio and
trajectory behavior; do not create targets by fuzzy-matching an ASR callsign to
an ADS-B callsign.

This is only a task hypothesis. The source paper notes that converting speech
to structured clearances and matching callsigns to timestamps are challenging
for TartanAviation, and that its single-airport coverage limits diversity.
[Published source/task discussion](https://link.springer.com/article/10.1007/s13272-025-00854-x).
It may prove too ambiguous or expensive to annotate; it is an aviation-domain
research task, not an operational ATC system.

The third-party Hugging Face derivatives are **not** accepted as gold labels.
Their cards describe ensemble ASR/ROVER labels and ADS-B callsign snapping;
the author says the pipeline fuzzy-matches spoken callsigns to ADS-B tails.
That makes snapped identity labels circular for a model receiving ADS-B. The
public dataset-server statistics endpoint reported a **partial preview** of
61,311 processed rows (`partial=true`), one airport, and 93 dates. Separately,
a pre-existing local metadata-only screen verified all 184 label shards
(25,246,122 bytes; 531,050 unique utterance IDs; 40,899 parent clip groups;
670 airport-date groups). It read only `utterance_id`, review/confidence and
agreement fields, plus the boolean `callsign_matched`; transcript and callsign
text columns were not read, and no media was downloaded. Of 531,050 rows, 157
were marked `verified` and 35 `edited` (192 rows across 191 parent clips); the
source has only a `train` split and no speaker ID. Those counts do not meet the
1,000 human-reviewed-example floor, and the 191 clips do not establish 200
independent groups. This full-shard metadata screen supersedes no official
source split and is not evidence of usable gold labels. Its existing local
artifacts are:

- `D:\CodexArtifacts\pretrained-reuse-r2\tartanaviation-atc-adbscreen-20261114-attempt-01\labels_metadata_integrity_summary.json` — SHA-256 `F8C1F8F19DDE8D845F208A9872E8F6479C46764C6CA4B5A11E30C619B5D56CF2`.
- `D:\CodexArtifacts\pretrained-reuse-r2\tartanaviation-atc-adbscreen-20261114-attempt-01\tartan_atc_labels_metadata_screen.json` — SHA-256 `B2AB25D4C9D0E0CD788C84F7036618B87E4853F5740A03947C5967877130CA1D`.
- Initial hash-only exposure roster for the official source and label derivative: `D:\CodexArtifacts\pretrained-reuse-r2\tartanaviation-atc-adbscreen-20261114-attempt-01\group_exposure_roster.json` — SHA-256 `82E71738FCB7199D387D12D01B03DBC169C5DC302130834770DC1E9D54DC2B3F` (preserved).
- Superseding roster for all 40,899 parent clips, keyed separately to the pinned official source, labels derivative, and utterance derivative revisions: `D:\CodexArtifacts\pretrained-reuse-r2\tartanaviation-atc-adbscreen-20261114-attempt-01\group_exposure_roster-v2.json` — SHA-256 `6426469A01127947190F570ED0776A77CBE61D73C7B59E44520500BEAFA4C9F3`. It contains no raw utterance or clip IDs. Reproduce with `scripts/hash_tartan_label_group_exposure.py`; this reads only the existing `utterance_id` column and rechecks every shard hash.

The dataset-server preview and the full local metadata screen are distinct
observations; neither admits this task. The group roster is ready for cohort-
ledger import, but the repository ledger has not yet been reconciled; fail
closed on any further use of this derivative or its groups. [Utterance card](https://huggingface.co/datasets/twangodev/tartanaviation-atc-adsb-utterances),
[label card](https://huggingface.co/datasets/twangodev/tartanaviation-atc-labels),
[author's pipeline description](https://twango.dev/writing/tartanaviation-atc-labels).

If a bounded payload/annotation feasibility pilot is later justified, it must
first establish source-exposure quarantine and group identities, then compare
audio-only, ADS-B-only, joint, missing-input, shuffled-trajectory, and
time-offset controls on held-out source groups. A weak or ambiguous human label,
fewer than 1,000 usable examples, or fewer than 200 independent groups after
identity grouping rejects the task before model fitting.

## Reproduction artifacts

- Audit code: `scripts/audit_tartanaviation_availability.py`
- Focused tests: `tests/test_tartanaviation_availability_audit.py` and `tests/test_tartan_label_group_exposure.py` (9 passed together).
- External result: `D:\CodexArtifacts\pretrained-reuse-r2\tartanaviation-atc-adsb-metadata-20261114-attempt-01\availability.json`
- Result SHA-256: `2c5a04e05017dd7a2a2627f6d25a9a0d1de068f950385cef8b073ca7a6265212`
- Executed script SHA-256: `ac0e6691b22e419f03676109382c02012685764f7b1a1c0da5b33969e6c2499a`
- Executed exposure-roster script SHA-256: `B7E36AA4EC9359E38B5B950E9F373685E803210D6792A72A25E63AF3300BEA19`.
- Nine focused tests pass; Ruff and `git diff --check` pass.

No model, audio, video, trajectory row, validation label, sealed audit, or paid
compute was used for this feasibility audit. R2 remains open; the next step is
to reconcile existing source exposure and identify a genuinely unexposed,
rights-cleared task with independently verified groups. Do not start a Tartan
payload pilot or training from the airport-date count alone.

# R0 live recheck — 2026-11-11

## Scope

Read-only refresh of repository, roadmap, PR, worktree, artifact, dataset-pointer,
process, GPU, and storage state before selecting the next R1/R2 action. The
latest roadmap source is Draft PR #15 at
`docs/pretrained-reuse-roadmap-20261010` / `e07e049ff4387484fb754e0fff963631f3cc7c85`;
it is not yet on `main`. This audit did not open dataset payloads, load a model,
run training/inference, or inspect sealed-audit data. No process or existing
artifact was changed.

## Live repository state

Snapshot time: **2026-11-11 15:41 JST**.

| Item | Observed state |
|---|---|
| `origin/main` | `09246d7b1821662a6b1eba47fc08c43f9dc3863c` |
| Roadmap PR #15 | Draft/Open; head `e07e049ff4387484fb754e0fff963631f3cc7c85`, base `main` |
| PR #15 checks | Both GitHub Actions `test` check-runs completed successfully on Nov 4 |
| Open PRs | 71 (#9–#79) from public GitHub REST API |
| Remote-tracking branches | 171 after `git fetch origin --prune` |
| Registered worktrees | 134 |
| Current user worktree | `codex/teacher-quality-clean-dev-v1` at `041c5954`; 9 changed paths, preserved |
| Other spot-checked worktrees | Path A shared-MiniLM, Path A video-text runtime, and AV-source screen clean; Perception MCQA worktree has 1 changed path and was left untouched |
| Global dirty-worktree count | **UNKNOWN**; only the listed worktrees were inspected for dirtiness |

The checked-out user worktree is on a historical Teacher branch, not the
roadmap branch. Its modified/untracked paths were not used as a base and remain
untouched. `main` still carries the older Gemma-first roadmap; R0–R8 direction
is taken from PR #15, as the objective requires.

## Current resources and active processes

| Resource | Observation |
|---|---|
| C: free space | 0 bytes |
| D: free space | 2,184,326,393,856 bytes at the initial check |
| RTX 3080 Laptop GPU | 16 GiB; 0 MiB allocated and 0% utilization at 15:30:40 JST |
| Matching Python/Torch/training processes | None observed in the process query |
| GPU-listed process | ChatGPT desktop UI only; no model process was listed |

Because C: is full, creating a regular worktree failed while Git attempted to
write administrative data below the C: repository. No cleanup was attempted.
This audit branch is isolated on D: in a shared-object clone; its commit will
contain only the new audit record. New large outputs must also go to D:.

## Branches, PRs, datasets, and artifacts

- The existing machine-readable `R0_INVENTORY.json` and checkpoint/source
  review `R1_AUDIT.md` are present in the PR #15 tree. Their original records
  were not rewritten. R1 already reports a successful local VideoPrism-B
  checkpoint/load path and retains V-JEPA 2.1 as REVIEW; rerunning the same
  load is not justified without a new decision question.
- Recent source-screen PR #69 did not admit a synchronized AV task. QuerYD and
  VoxDub fail rights/task-fit gates; the concert dataset is non-commercial and
  too small. PR #74 keeps otoSpeech unadmitted: only 58 sessions are reported,
  below the provisional 200-independent-group floor, and its task conversion
  and access remain unresolved. These are metadata-only decisions; no gated
  payload was accessed.
- The frozen-data/manifests policy and earlier task-specific results remain in
  PR #15 and the independent PR branches. This audit did not reopen their
  validation or sealed-audit samples. Candidate availability is therefore
  tracked from the source manifests/reviews, not by treating old evaluation
  artifacts as fresh data.
- Artifact roots checked: `C:\CodexArtifacts\pretrained-reuse-r2` (2,631
  top-level entries) and `D:\CodexArtifacts\pretrained-reuse-r2` (65
  top-level entries). They were not modified. The tested canonical
  `CodexDatasets\pretrained-reuse` paths were absent; this does not prove that
  no dataset copies exist elsewhere.
- A D: artifact at
  `D:\CodexArtifacts\pretrained-reuse-r2\path-a-unified-runtime-smoke-20261111-attempt-04\run-metadata.json`
  records a 7-request integrated smoke: 5 succeeded, 2 returned Windows
  `OSError: [Errno 22] Invalid argument` before a task ID was assigned. The
  package records 192,709,534 unique parameters, 731,862,633 payload bytes,
  `supports_joint_modalities: false`, and no quality evaluation/training. Its
  named source commit `025336e40aeeda2db32d459703beb3342d519395` is not present
  in the current local/fetched Git object database, so the artifact is not
  reproducible from current refs until the code revision is recovered. A
  read-only GitHub commit lookup also returned HTTP 422, `No commit found for
  SHA`, confirming it is unavailable from the repository API. The
  recorded package is above the provisional 200 MB size target.

## Constraints and decision

- Preserve all 134 worktrees, open PRs, data, and artifacts; do not clear C: by
  deleting experiment outputs.
- Do not use paid compute. Do not interrupt live work. No live model work was
  detected at this snapshot.
- Do not treat independent modality smoke routes as joint multimodal use.
- Keep R0 device/task acceptance and R2 informative joint-AV quality gates
  open. Existing evidence does not support R3 shared-backbone or weight-merging
  experiments yet.

The immediate next step is to continue from the pinned R1/R2 records on an
isolated D: checkout, recover or replace only the unreproducible runtime
candidate after inspecting the newest Path A branches, and select the next
experiment from a rights-cleared, exposure-audited task. The current lack of an
admitted fresh AV task is the active research constraint, not GPU availability.

## Same-day follow-up and experiment triage

Snapshot time: **2026-11-11 15:58 JST**. The current user worktree remains
unchanged, including all nine pre-existing changed paths. The local RTX 3080
again showed 0 MiB / 0% use, with only the ChatGPT desktop process listed by
`nvidia-smi`. C: now has 2.43 GiB free and D: has 2,031.92 GiB free; use D: for
any new large data or model outputs. No training process was started.

Two existing result lines were checked before choosing another action:

- The fixed-feature iNaturalist readout from 2026-10-31 already used 1,145
  training images and 430 observer-disjoint development images. Its 49,793
  parameter scorer reached 75.35% Accuracy, with 98.17% train Accuracy and a
  22.82-point gap. Accuracy gain over the prior readout was only 0.47 points
  with a paired observer-cluster interval spanning zero; its NLL/Brier/ECE
  improved. That validation was already involved in earlier selection, so it
  is not eligible for more tuning.
- The existing RoboMNIST capacity-CV artifact on D: selected rank 8 using
  train-only grouped five-fold selection. On the already observed development
  validation, its Accuracy was 49.75% versus 53.35% for the preserved rank-32
  reference; its paired Accuracy interval against audio-only included zero.
  This does not support more RoboMNIST fitting or a claim of useful joint-input
  quality. The original 2,000-row validation remains development evidence,
  not blind evidence.

A proposed newer iNaturalist metadata cohort was compared by record identity
against one merged feature cache, which was an incomplete history check. The
initial interpretation of its validation as fresh is superseded by the
correction below. No media download, feature extraction, or fitting was
started. Frozen metadata is at
`C:\CodexArtifacts\inat-birds-cc0-expansion-filtered-pages-11-30-20261029-attempt-02`;
the train, validation, and metadata SHA-256 values are respectively
`362a562568910676a15b71d42233fbeb8f6811116afcb6ef22f846bcf83e8618`,
`c3f846c677516b5f08c82f6ea12a39146bf9a890be66306985f202622d6ddcc0`, and
`8e1c8e65c281018657caa3edc8c2f0374d934b34a59fe949deba9baa9088ba22`.

The user's uncommitted iNaturalist downloader was reviewed read-only and left
untouched. Before it can support that cohort, its reported near-duplicate
hashes must be made part of the actual cross-split pass/fail gate; the current
`media_identity_gate_pass` checks exact hashes only. This is an implementation
integrity fix, not a reason to weaken the split.

**Triage:** do not repeat the exposed iNaturalist or RoboMNIST experiments.
Keep R2-M data/task admission as the blocking gate. Continue only with a
rights-cleared and exposure-audited task where both inputs can be tested with
predeclared ablations and independent source groups; do not begin R3 shared
backbone/weight merging until that gate passes. The alternate VideoMAE route
may be developed as a separate Path A runtime, but it needs its own frozen
feature cache and task-specific evaluation and cannot be presented as a
VideoPrism comparison.

## Targeted synchronized-source follow-up — 2026-11-11 16:11 JST

An additional official-source screen did not find a new R2-M admission:

| Source | Official facts | Disposition |
|---|---|---|
| [Action Verb Corpus](https://www.ofai.at/resources/avc) | CC BY 4.0; 46 recordings from 12 people, with audio, video, hand/arm motion, and 390 simple action instances. | Too few independent people/recordings for the existing minimum; no fit. |
| [XRFv2 Plus](https://airslab2020.github.io/labpages/datasets/xrfv2-plus/index.html) | 853 synchronized sequences from 16 volunteers; 108,929 QA pairs do not increase the number of independent groups. | Below the 1,000-example / 200-group admission floor, and the tracked source/rights screen remains unresolved; no acquisition. |
| [Argoverse 2 Sensor](https://www.argoverse.org/av2.html) | 1,000 camera/LiDAR scenarios, but the official data license is CC BY-NC-SA 4.0. | Exclude for this product-directed workstream; the task is also camera/LiDAR rather than audio/video. |
| [Seeing Sound](https://github.com/sonyc-project/seeing-sound-dataset) | CC BY 4.0 but only 60 synthesized audio soundscapes; the visualization is annotation assistance, not a synchronized video input. | Not an AV task or sufficiently independent source set. |

No candidate in this screen passes the existing task, rights, group-count,
and both-input gates. No data was downloaded, no media or labels were opened,
and no model was run. Continue R2-M source discovery; do not substitute an
easy single-stream task or change the group floor to manufacture a pass.

## Correction — iNaturalist validation exposure — 2026-11-11 16:17 JST

The earlier same-day follow-up incorrectly treated the 446-row validation
manifest as unseen because it compared only with the merged feature cache
containing a different 430-row validation split. A read-only scan of all
`C:\CodexArtifacts\inat-birds*` JSONL manifests found that the full 446-row
validation is already present in the earlier source manifest and feature run:

- Proposed `attempt-02/validation.jsonl` and prior
  `attempt-01/validation.jsonl` are byte-identical (SHA-256
  `c3f846c677516b5f08c82f6ea12a39146bf9a890be66306985f202622d6ddcc0`).
- The matching validation records occur in the prior 1,367-row VideoPrism
  feature cache; its `feature-rows.jsonl` SHA-256 is
  `d772a67381f37dbda25c77d7a2ee4e841773573023dfa920082e4f4be1d309ae`.
- The prior linear-readout run has 446 validation predictions with SHA-256
  `0fc854c25a8d9887674c12fb1b6af50a97992baf886d9bbdfd2f3b89b490fbeb`; its
  result record SHA-256 is
  `bbc5f4d03b0c6fc0c5242a6f7e1f5cb324fe08e84522eec631b0555def49a56f`.
- All 446 proposed validation observation IDs, photo IDs, and observer hashes
  occur in the prior validation manifest and feature rows. The prior
  prediction file covers the same 446 observations and photos in the same
  order.

Therefore this validation is exposed development data and is closed to further
model selection or quality claims. The exact match demonstrates duplicate
records, not an independent fresh cohort. The prior report's claim that the
new validation could provide independent evidence is withdrawn. The iNaturalist
run was not launched; no existing artifact or user work was altered. Any future
use requires a genuinely new, adequately sized cohort and a source-global
exposure audit before acquisition or model loading.

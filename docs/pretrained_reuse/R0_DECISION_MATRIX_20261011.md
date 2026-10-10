# R0 candidate matrix and execution contracts (2026-10-11)

Status: evidence and proposal for review. This document does **not** record
approval of product size, deployment device, task thresholds, or time/money
limits. Those choices remain a human decision in `ROADMAP.md` before any
training or promotion decision.

## Candidate matrix

| Candidate/path | Evidence available | Fit and unresolved gate | Next permitted action |
|---|---|---|---|
| VideoPrism-B image/video | Pinned official checkpoint passes rights and local 8-frame forward gates; 114,365,184 parameters. | CLEVRER development probe is weak and not a blind task-quality result; full package and end-user runtime are unmeasured. | Continue bounded frozen-feature Path A evaluation using the existing protocol. |
| V-JEPA 2.1 ViT-B image/video | Pinned 1,664,223,428-byte checkpoint strictly loaded and image/video forward audited. | Exact checkpoint grant remains REVIEW; do not package or promote. | Keep as research comparison only until exact weight terms are explicit. |
| Ruri v3-30m Japanese text | Pinned small checkpoint loaded; one zero-shot Japanese diagnostic is near chance. | Training-data contamination risk; the diagnostic is not a Decision-head result. | Keep separate from product claims; require task-matched, rights-cleared validation. |
| MiniLM-L3 English text | Pinned Apache-2.0 checkpoint loaded; English text feature path works. | Synthetic text probe only; no factual English Decision quality established. | Continue the fixed Path A synthetic probe, labeled synthetic. |
| Whisper-tiny speech | Pinned Apache-2.0 encoder loaded; feature plumbing works. | Speech Commands task quality not established; current checkpoint is an ASR encoder. | Use only for task-matched frozen keyword evaluation. |
| EfficientAT DyMN-0.4 AudioSet | Pinned 8,086,973-byte release strictly loaded; owner explicitly permits commercial checkpoint use. | Environmental audio tagging only; no speech keyword or joint AV result. | ALLOW as a small environmental-audio candidate; do not substitute its task for speech. |
| YT-NTU-AVQ joint AV quality | Pinned, ungated 1,620-sequence dataset with MOS annotations; metadata and archive hashes recorded in the candidate manifest. | CC BY 3.0 terms are explicit, but the local license allowlist returns REVIEW and original source IDs are missing from metadata; perceptual quality is not semantic AV reasoning. | Review-only; no corpus inclusion, media fetch, or readout training yet. |
| PE Core T/S/B | Official docs list T/16, S/16, B/16 checkpoints and Apache-2.0 checkpoint terms. | The implementation package's `setup.py` declares FAIR Noncommercial Research; the anonymous Hub API audit returned 401, so local artifact access/load was not verified. | Keep rights/access split explicit; no product use of the official implementation without resolution. |
| ViT-Lens-L | Pinned released CLIP ViT-L/14 + audio Lens pair totals 12,326,294,229 bytes. | Too large for the proposed compact path; original backbone, not V-JEPA; no matched Decision result. | Reference only; no local download or load planned. |
| VATT | Official README lists Google Storage checkpoints. | Tested link redirects to sign-in; exact weights, hash, license, and current runtime unresolved. | UNKNOWN; revisit only if anonymous access and exact terms are established. |
| MJEPA | Official MIT code and training/evaluation configs; no official release assets found. | No pretrained checkpoint available for local weights audit; published configs use large backbones. | Architecture reference only. |

This matrix separates checkpoint rights, source-code rights, availability,
runtime feasibility, and task quality. One passing column does not imply the
others pass. Detailed pinned evidence is in `R1_CANDIDATE_INVENTORY.json`,
`R1_AUDIT.md`, and `R1_RELEASE_LICENSE_FOLLOWUP_20261011.md`.

## Evaluation plan

Use `EVAL_PROTOCOL.md` as the frozen protocol for Path A. Each task returns a
probability distribution over the supplied choices. Report Accuracy, NLL,
Brier, ECE, group counts, and source/modality breakdowns; use the protocol's
fixed group-paired bootstrap. Preserve sample order and IDs, and group splits
by the strongest underlying identity (source/state, image, audio speaker/source,
or video scene). Development validation supports iteration; sealed audit stays
unloaded until the architecture, preprocessing, calibration, selection rule,
and thresholds are frozen. Do not combine separate unimodal scores into a
joint audio-video capability claim: there is no currently approved synchronized
AV Decision source in the manifests.

The current executable work remains frozen-feature readout evaluation, not
backbone training, unfreezing, distillation, or product promotion. The existing
historical `>=95%` aspiration is a challenge target only; actual task thresholds
and acceptance bounds require user approval before they can govern a training
or promotion decision.

## Separate environment contracts

| Profile | Purpose | Contract |
|---|---|---|
| CI / CPU | Unit tests, lint, manifest and catalog validation. | Python 3.11; no model downloads, GPU requirements, or training. |
| Existing local CUDA audit | Small, read-only pinned checkpoint load/forward and bounded frozen-feature experiments on the RTX 3080 Laptop GPU (16 GiB). | Reuse the already working isolated Python 3.11 / PyTorch 2.6.0+cu124 environment. Record exact package versions and hashes per run. No package upgrades, paid compute, or large model test while another GPU job is active. |
| TensorFlow / legacy candidates | VATT or other incompatible legacy stack, if ever justified. | Separate environment and lockfile; do not alter the active Teacher/QAT or PyTorch environment. VATT's documented TF/CUDA stack is not currently validated locally. |
| Model/source audits | PE, EfficientAT, or other upstream code with distinct licensing/dependency needs. | Isolated checkout and dependency directory under `C:\CodexArtifacts`; pinned revision and asset hashes; do not install into the project/global Python environment. |
| Artifacts and datasets | Checkpoints, media, feature caches, predictions. | Store outside Git under the dedicated artifact root. Preserve existing corpus, checkpoints, worktrees, runs, and PRs read-only unless a later explicit work item says otherwise. |
| Cloud | None for current R0/R1/R2 work. | $0 cloud spend. Any future paid compute requires a separate approved decision. |

R0 approval remains open for: target deployment device, maximum installed
artifact size, per-task quality floors and confidence bounds, and allowed
elapsed time/cost. This file makes the candidate and execution constraints
reviewable without silently setting those product decisions.

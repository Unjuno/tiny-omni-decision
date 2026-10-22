# R0 candidate matrix and execution contracts (2026-10-11)

Status: evidence and proposal for review. The user authorized the ROADMAP's
100–200M parameter and <=200 MB package figures as provisional engineering
targets. This does **not** approve a deployment device, task thresholds,
confidence bounds, or time/money limits; those remain open before any product
selection or promotion decision.

## Candidate matrix

| Candidate/path | Evidence available | Fit and unresolved gate | Next permitted action |
|---|---|---|---|
| VideoPrism-B image/video | Pinned official checkpoint passes rights and local 8-frame forward gates; 114,365,184 parameters. | CLEVRER development probe is weak and not a blind task-quality result; full package and end-user runtime are unmeasured. | Continue bounded frozen-feature Path A evaluation using the existing protocol. |
| V-JEPA 2.1 ViT-B image/video | Pinned 1,664,223,428-byte checkpoint strictly loaded and image/video forward audited. | Exact checkpoint grant remains REVIEW; do not package or promote. | Keep as research comparison only until exact weight terms are explicit. |
| Ruri v3-30m Japanese text | Pinned small checkpoint loaded; one zero-shot Japanese diagnostic is near chance. | Training-data contamination risk; the diagnostic is not a Decision-head result. | Keep separate from product claims; require task-matched, rights-cleared validation. |
| MiniLM-L3 English text | Pinned Apache-2.0 checkpoint loaded; English text feature path works. | Synthetic text probe only; no factual English Decision quality established. | Continue the fixed Path A synthetic probe, labeled synthetic. |
| Whisper-tiny speech | Pinned Apache-2.0 encoder loaded; feature plumbing works. | Speech Commands task quality not established; current checkpoint is an ASR encoder. | Use only for task-matched frozen keyword evaluation. |
| EfficientAT DyMN-0.4 AudioSet | Pinned 8,086,973-byte release strictly loaded; owner explicitly permits commercial checkpoint use. | Environmental audio tagging only; no speech keyword or joint AV result. | ALLOW as a small environmental-audio candidate; do not substitute its task for speech. |
| FAIR-Play synchronized AV | Dataset-level CC BY 4.0 is stated by the pinned upstream README and verified against its exact LICENSE file. Eight train-only pairs were previously preflighted. | Original scene IDs are absent from published split lists; task/hard-negative design and proof of both-input necessity are not established. Full corpus is ~100 GB; current C: capacity is insufficient. | Research candidate for bounded train-only task construction only; no quality-selected fitting until grouping, task controls and capacity preflight pass. See `R2_FAIRPLAY_LICENSE_AND_GROUP_FOLLOWUP_20261022.md`. |
| YT-NTU-AVQ joint AV quality | Pinned, ungated 1,620-sequence dataset with MOS annotations; metadata and archive hashes recorded in the candidate manifest. | CC BY 3.0 terms are explicit, but the local license allowlist returns REVIEW and original source IDs are missing from metadata; perceptual quality is not semantic AV reasoning. | Review-only; no corpus inclusion, media fetch, or readout training yet. |
| UnAV-100 joint AV event localization | Official page describes >10K untrimmed videos, >30K events, 100 categories, temporal boundaries, and a YouTube-ID/timestamp/split CSV; project page states CC BY 4.0. | Official page reserves original video-owner copyright; HF mirror advertises conflicting CC BY-NC-SA 4.0. Source-level rights, immutable data revision, archive hashes, and split identity are unverified. | Review-only; exclude the HF mirror and do not fetch/use assets until license scope, per-source rights and grouping are reconciled. |
| CREMA-D synchronized AV acted-affect decisions | Official repository at `1658cd342dff90010aa843eaeebd53610a08b1dc` describes 7,442 clips from 91 actors, crowd-rated audio-only/video-only/AV emotion and intensity, ODbL 1.0 database + DbCL 1.0 contents, and a user access form. | Narrow acted-affect task does not address general AV event reasoning; identifiable voice/video and actor metadata need component rights/privacy review; not in approved manifests. | Review-only, task-mismatched; no access form submission, media fetch, manifest admission, or model fitting. See `R2_CREMA_D_AV_SOURCE_SCREEN_20261011.md`. |
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
joint audio-video capability claim: YT-NTU-AVQ and UnAV-100 are review-status
candidates, and no synchronized AV Decision source is approved in the manifests.

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

## Addendum — provisional engineering contract (2026-10-18)

The user confirmed the roadmap parameter/package targets as provisional. For
planning and experiment reporting, apply them to the **complete pre-quantized
inference system**: count every resident encoder, tokenizer/embedding,
projection, fusion/readout, normalization and adapter parameter. Do not count
parameter reduction from quantization toward the pre-quantization cap. Record
post-quantization package bytes separately; the provisional package target is
<=200 MB including weights, scales, exceptions, tokenizer/preprocessing assets,
metadata and runtime-required adapters.

The current Windows 11 / RTX 3080 Laptop (16 GiB) machine is an **engineering
reference host only**, not a selected end-user deployment device. Experiments
must record this host and actual resource use; do not infer mobile/edge latency,
RAM, energy or thermals from it. The user-authorized local-compute policy is
cloud spend $0 and no hardware purchases. User-approved cooperative yielding of
the Orbitune GPU workload permits a ready, justified local training job to
request its normal yield behavior; it does not justify starting a GPU job before
data, safety and acceptance preflights pass.

For each predeclared required task, keep >=95% Accuracy as the required product
goal and 98% as the aspiration, never as a pooled four-modality substitute.
These are not yet a release pass: the task battery, sample/group counts,
independent test protocol, confidence-bound rule and risk-specific abstention
policy have not been frozen. Report task-wise Accuracy, NLL, Brier and ECE;
also report group-aware uncertainty and coverage when abstention is used. Do
not select on sealed-audit results. For joint tasks, a task is eligible only
when labels establish which information is sufficient/necessary; input
sensitivity alone is not such evidence.

Before every fit, preregister a bounded wall-time and output-space budget from
the measured dry run. Record elapsed wall time, GPU/CPU/RAM/VRAM, output bytes,
and local energy only if measured. A universal latency, energy or elapsed-time
release threshold remains **unset** until the end-user device and named tasks
are selected. Retain p50/p95 cold and warm end-to-end latency measurements as
engineering evidence, with cache-hit and encoder-only timings clearly
separated.

Thus R0 is **partly specified, not closed**: provisional size, local-only/$0
compute, current reference host, and per-task aspiration are recorded; final
device, task battery, independent-test confidence bounds, calibration policy,
latency/energy limits and release thresholds remain open. No product pass is
implied by this addendum.

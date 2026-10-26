# Perit duplex support audio temporal-task screen — 2026-10-26

## Decision

**Source rights and real audio forward pass verified; not admitted to R2-M2 or
training.** The public Perit sample is a plausible narrow audio-only turn-taking
task, but it has no video stream and only four independent performer-pair
groups. It cannot satisfy the synchronized AV gate or support a broad audio
quality claim. No model fitting, checkpoint selection, full-corpus download,
or sealed evaluation was performed.

The earlier project decision withdrew a separately scoped small audio-only
training/product effort. This source screen does not reopen that effort. If a
future M1-A temporal probe is explicitly included as a component of the full
Omni path, its target and evaluation protocol must first be admitted against
the current R2-M protocol.

## Pinned source and rights

- Dataset: `perit-ai/perit-duplex-support-en`
- Hugging Face revision: `2b280ef137530d682212d33916c3b6986d0d72ab`
- The publisher describes 45 public recordings totaling 5.1 hours, 48 kHz,
  stereo 16-bit PCM, with a different speaker on each channel and overlaps
  retained. The open sample is CC BY 4.0. The publisher states that performers
  consented to publication and training. The remaining 10.4 hours are available
  only on request and were not requested or accessed.
- Content is scripted English customer-support role-play, not spontaneous
  conversation. The eight performers form four recurring, disjoint pairs.
- Source links: [publisher dataset page](https://perit.ai/datasets),
  [pinned dataset revision](https://huggingface.co/datasets/perit-ai/perit-duplex-support-en/tree/2b280ef137530d682212d33916c3b6986d0d72ab).

Downloaded metadata-only files are outside Git at
`C:\CodexArtifacts\pretrained-reuse-r2\perit-duplex-turntaking-source-screen-20261026`:

| File | SHA-256 |
|---|---|
| `README.md` | `7988ecc6149a8b11a45ff6b82d49eff026c2d00886c4e12e1acaac95b4af9aae` |
| `metadata.csv` | `9b7d2892a0b84da37826fe94ca98c8244fd80890f71e7d837ed902c312e3a0e9` |

The metadata audit found 45 unique recording names, 5.0981 total hours, and
four pair groups: Aldo/Michael 15 recordings, Andy/Dom 14, Amanda/Heather 9,
and Dylan/Jonathan 7. There are eight listed session IDs (several within a
pair), but the independent performer-pair count remains four. A pair-held-out
fold would therefore have only one held-out pair and three training pairs.

## Real-media runtime smoke

Four short files (one per pair) were downloaded from the pinned revision. Their
names, byte lengths, and SHA-256 values are in
`audio-smoke-receipt.json` beside the metadata. On the exact local
EfficientAT DyMN-0.4 checkpoint from R1 (1,961,239 parameters,
SHA-256 `49b0e63577021994257b3de2ac5f1686d71554b93c8149f9d2568ad5760b904e`),
strict state loading and CPU forward over eight real one-second channel
windows succeeded. The output shapes were `[8, 527]` logits and `[8, 384]`
pooled features; all values were finite. The run took 0.452 seconds on the
local CPU with the already isolated PyTorch/torchaudio environment. Full
receipt: `efficientat-real-audio-smoke.json`, SHA-256
`c22d5d1d79ecfe75f5c285f3da02cbe7a44fef5d0deb89f98f07d4eff9711fe9`.

This proves real-input loading/preprocessing/forward feasibility only. It is
not a Decision metric or evidence that EfficientAT features predict turn
choices.

## Label and task limits

The official metadata exposes each channel's first speech onset, not dense
segment-level labels. A non-training audit compared 20 ms per-channel RMS
activity, separated by a per-record Otsu threshold, with the onset metadata.
Several first-detected spans were early because short low-energy spans/noise
preceded the marked onset; in one file a 20 ms transient appeared about eight
seconds before the marked onset, while the first sustained span began within
0.04 seconds of it. This threshold exercise is not an annotation-quality
validation and its generated activity labels must not be treated as ground
truth. The publisher states that channel-level voice activity is available by
construction, but a causal, reproducible labeling rule still needs to be
specified and checked over the complete open set before a turn-prediction
target is used.

A possible future text-free task is “which channel will speak in the next
fixed interval?” using only preceding audio, with typed left/right options.
This is a conversational temporal task, not environmental-event recognition
and not AV fusion. Before fitting, it would need a frozen causal label rule,
pair-held-out split, fixed persistence/prior baselines, and a predeclared
window/overlap policy. Four pair groups support at most an exploratory
cross-validation estimate with substantial group uncertainty; they do not
support a 95–98% general audio claim. The full 5.1-hour corpus was not
downloaded.

## Reproduction artifacts

External artifacts and the four-file audio receipt are under the path above.
The EfficientAT forward used the pinned checkpoint and source already audited
in `R1_FROZEN_COMPONENT_CPU_RECHECK_20261026.md`; no repository dependency or
existing model artifact was changed. The worktree was separate from other
active user changes.

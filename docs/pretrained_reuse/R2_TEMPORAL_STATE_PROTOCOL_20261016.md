# Temporal state candidate: bounded connection experiment

## Intent and decision before execution

The user authorizes LSTM/xLSTM and other temporal architectures under the same
<=200M **prequantization total** and conditional Omni Decision objective. This
is an architectural extension of Path A, not a replacement of its product
scope by a fixed classification task. Autonomous design/implementation is
authorized in the conversation; the existing baseline and artifacts remain
read-only. This protocol specifies the first bounded implementation and probe.

Current `NativeCandidateReadout` uses one cross-attention and has no recurrent
state. A separate earlier Lombard experiment used GRU64: temporal visual input
helped video-only prediction, but joint answers ignored mismatched video. That
result does not establish general temporal or joint reasoning.

Implement an **optional, separate LSTM residual adapter**, with frozen native
encoders and the unchanged candidate readout. Each video spatial cell follows
its own chronological sequence; shared weights process these sequences in
parallel. Speech follows its original token timeline. Do not serialize spatial
neighbors as successive moments, discard the spatial grid, or collapse all
modalities into a single final vector. Text/image/global event-audio tokens
pass through unchanged. The readout can access every residual-enhanced token.

Each temporal path: Linear(D,192), LayerNorm192, unidirectional LSTM192,
Linear(192,D,bias=False), residual addition. Zero-initialize only the last
projection: the new candidate starts exactly at the existing readout function.
Initially only that projection receives nonzero adapter gradients; after an
update the recurrent path must receive them too. This is deliberate, not a
claim that all parameters learn on update one. Count every added parameter.

Use explicit caller-owned LSTM state in the low-level sequence primitive;
never persist it implicitly across examples. The native wrapper resets for
each input block. Validate finite aligned shapes, devices, masks and ordered
timestamps. Masked steps neither update memory nor affect valid outputs.
Spatial identities must repeat across video frames; reject malformed layouts.
This is fixed-grid memory, **not object tracking**. Elapsed time is retained in
the original coordinates, not separately modeled by a continuous-time cell.

## Alternatives and selection

1. Existing attention readout: required baseline, zero integrated fit updates.
2. Residual LSTM: first executable temporal candidate; dependency already exists.
3. xLSTM: subsequent alternative, not yet implemented or quality-tested here.
   sLSTM is relevant to state updating; mLSTM to associative memory. Compare
   actual parameter/compute budgets rather than assuming either is superior.

Primary references, read 2026-10-16:
[xLSTM paper](https://arxiv.org/abs/2405.04517),
[official implementation](https://github.com/NX-AI/xlstm),
[sLSTM backend code](https://github.com/NX-AI/xlstm/blob/main/xlstm/blocks/slstm/cell.py).
Official code exposes a vanilla PyTorch sLSTM backend and native alternatives
for the larger implementation. CUDA kernels have additional platform/build
requirements. Do not upgrade the working Windows environment or claim local
xLSTM runtime compatibility before a pinned independent probe. No xLSTM model,
package or kernel is installed in this first unit.

## First probe, fixed before running

CPU tests: unchanged baseline at initialization; chunk versus whole sequence;
prefix independence from future **feature** tokens; mask-hole independence;
explicit state/reset; spatial-cell identity; rejection of duplicate/unordered
coordinates; candidate-order equivariance; gradients after two diagnostic
updates; checkpoint reload; <=200M accounting.

Reuse the previously hashed TRAIN-only native integration arrays and queries.
No labels, validation, calibration, or audit are loaded. Execute on local GPU
if idle: compare baseline and LSTM initialization, reverse candidate ordering,
apply two **throwaway gradient plumbing updates with an artificial target**,
then check nonzero finite gradients, save/reload, repeated inference and wall
time/peak memory. Those two updates are NOT task training, quality evaluation
or a candidate to select. Save them separately with this limitation. Use two
warmup calls and ten timed calls, synchronize CUDA around timings. Report
head-only timing, excluding frozen encoders, as such. New output directory only.
Stop on nonfinite values, source/hash changes, or total parameters >200M.

Existing full-clip encoders can use future frames/samples. A causal LSTM head
does **not** make the full model causal or streaming. Independent relative
timelines do **not** establish audiovisual synchronization. A global event-audio
vector does not expose event timing. These remain explicit product limitations.

## Quality experiment remains required

Finish the common task/query mapping and freeze sampler, losses, train budget,
selector and retention gates. Compare baseline versus this optional path on
identical data/seed/order/updates, while also reporting additional compute and
parameters. Include a parameter-matched nonrecurrent residual control before
attributing a gain specifically to recurrence. No broad architecture sweep.
Report all modalities, temporal tasks, condition pairs, joint input mismatch,
NLL/Brier/ECE, min task, resource cost and uncertainty. A feature reversal is
an internal diagnostic, not a fresh raw-video causal benchmark. If recurrent
memory brings no reproducible gain, reject it; a small useful LSTM may be kept
without xLSTM. If memory is justified but limited, audit a pinned xLSTM variant
as the next one-change alternative. Do not treat this smoke as a quality gate.

## Implementation sequence

Write regression tests first; add a standalone module without changing the
baseline file; run focused and full CPU checks; commit the executable source;
run the isolated physical-feature probe and fresh-process replay; preserve
receipts; record measured costs and remaining quality work; commit/push results.

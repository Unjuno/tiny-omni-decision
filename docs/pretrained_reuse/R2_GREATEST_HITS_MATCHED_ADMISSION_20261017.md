# Matched AV admission: complete inputs and five full-path smokes

**Input audit and all five smokes PASS. Full learning curves remain pending.**
Execution source is `8a5440da334e74e970d80a7059ca5bb99a7248cb`. These results admit the fixed
[matched protocol](R2_GREATEST_HITS_MATCHED_LEARNING_PROTOCOL_20261017.md); they
supply mechanics, reload and local cost evidence. The 32-update smoke scores are
not used to rank architectures, select hyperparameters or claim joint quality.

## Complete input evidence

The original CPU video producer completed normally: 11,151 events, 409 parents,
32,582 development questions and 21 matched smoke events. Frozen encoder state
before/after agrees. Its terminal status is `PASS_COMPLETE_VIDEO_CACHE` and wall time is
13655.7 seconds. The earlier independently revalidated audio remains
admitted through its explicit pinned receipt; its original failed producer receipt
is preserved.

The separate CPU input audit took 293.1 seconds and passed all
11,151 video events, actual clock/feature bindings and 21 exact smoke references,
alongside complete audio/query/common rehearsal input admission. Canonical input
digest: `00aa90bed282446e77ae834afc4d9b1d1c1adeedcbaf73e47a49f084c1838868`. Hash/identity/shape/clock integrity does not establish
adequate semantic labels, physical AV synchronization or actor/site independence.
No optimizer or encoder update was performed in this audit.

## Full-path smoke measurements

Each arm used the same pinned data, fitted initial model, exact sampling prefix
and 32 optimizer updates, then evaluated all six full TRAIN/validation views.
Saving and reloading the checkpoint reproduced every saved score/gate matrix
exactly. All five terminal statuses are `PASS_MATCHED_AV_SMOKE` with no error.

| Arm | Encoder-inclusive parameters | Smoke wall seconds | Peak CUDA allocation bytes | Projected full hours |
|---|---:|---:|---:|---:|
| C0 | 196,254,874 | 1601.1 | 272,823,808 | 9.30 |
| E0G0 | 197,097,949 | 1753.2 | 292,531,712 | 9.92 |
| E0G1 | 197,097,949 | 1769.2 | 293,273,600 | 10.10 |
| E1G0 | 197,097,949 | 1809.6 | 296,104,448 | 9.88 |
| E1G1 | 197,097,949 | 1855.7 | 296,846,336 | 10.09 |

Hardware: local RTX 3080 Laptop, 16,384 MiB. These CUDA allocation peaks exclude
host RAM, driver reservations and other applications. Full projections combine
measured update time, setup and twelve full evaluation/reload passes; they are
estimates from short timing samples, not completed training times. Each arm's
exact backend, versions, config, per-update timing and all output hashes remain
in its external receipt. The training allowance must cover at least 1.5 times
that arm's projection; the current C0 allowance is 72,000 seconds.

## Launch outcomes and actual boundary

All attempted launches remain recorded. The first C0 launch failed at import
because the repo was absent from `PYTHONPATH`. After correcting only that launch
environment, C0 attempt 01 failed before optimizer updates because `--reference`
expects a smoke **directory**, not its `result.json` file. The corrected command
uses the directory with the SHA-256 of that directory's receipt.

C0 attempt 02 reached initial step-0 evaluation but an unrelated GPU training
process appeared after launch. Only this newly launched C0 process was stopped,
with zero optimizer updates; its partial predictions and `interruption.json`
remain preserved. No external process was stopped. This incomplete attempt is
neither a quality failure nor a fully resumable training checkpoint.

After the other GPU job ended and an authoritative process/GPU check confirmed
idle hardware, C0 attempt 03 launched on 2026-10-17 at 10:03:49 JST (PID 1320).
At this report's snapshot it is RUNNING in initial evaluation, with zero optimizer
updates; this is a historical observation, not a promise that the PID stays live.
Inspect the real process and receipt before continuing. No other matched arm has
started full training. The runner does not implement interrupted-run resume.

The full train command is:

```powershell
$env:PYTHONPATH='C:/Users/junny/.codex/worktrees/r2-av-transport/tiny-omni-decision'
& 'C:/Users/junny/AppData/Local/Programs/Python/Python311/python.exe' `
  'C:/Users/junny/.codex/worktrees/r2-av-transport/tiny-omni-decision/scripts/train_greatest_hits_matched.py' `
  --phase train --arm C0 `
  --output C:/CodexArtifacts/pretrained-reuse-r2/greatest-hits-matched-train-C0-03 `
  --audit C:/CodexArtifacts/pretrained-reuse-r2/greatest-hits-matched-input-audit-01 `
  --audit-sha256 69a489b4fa4c81ed155d83328b816409326666f587add8392ea21a4e704efdad `
  --reference C:/CodexArtifacts/pretrained-reuse-r2/greatest-hits-matched-smoke-C0-01 `
  --reference-sha256 946ac1bfbc77904f40f535016aedc1362fd82d712f8c0783b11fb60bbf70c104 --max-seconds 72000
```

It starts from the protocol's fitted step-8192 model with a fresh optimizer.
This is model warm start, not random initialization or an exact training resume.
The smoke-trained 32-update checkpoint is only an admission reference, not the
initial model for full training.

## Evidence and verification

External root: `C:/CodexArtifacts/pretrained-reuse-r2/`. Complete input admission
is `greatest-hits-matched-input-audit-01/`; smokes are
`greatest-hits-matched-smoke-<arm>-01/`. The compact independently rechecked summary
is `greatest-hits-matched-admission-summary-01/result.json`. The repository
[manifest](../../manifests/pretrained_reuse/greatest-hits-matched-admission-20261017.json)
stores exact paths, config hashes, every smoke result hash and failed launches.
Thirty CPU runner/runtime tests passed in 5.72 seconds in the separate evidence
worktree with CUDA hidden. No runtime implementation or training source changed.

| Evidence | SHA-256 |
|---|---|
| Full video result | `9fdb5bfcf75a35c73d3b3634e1affbeb0373074b8cc9240f6567548aba8922ce` |
| Independent input audit result | `69a489b4fa4c81ed155d83328b816409326666f587add8392ea21a4e704efdad` |
| Input admission | `b62adbe352752d1d8b376414210d906f2493187c629e4da9a2dbbad9cf6dbe30` |
| C0 smoke result | `946ac1bfbc77904f40f535016aedc1362fd82d712f8c0783b11fb60bbf70c104` |
| E0G0 smoke result | `e5fe6fdaec7c35ab1fc42005fa62e81beb8d743405c0d7fcd8cfe8d3fe824645` |
| E0G1 smoke result | `4312cb224fd88115807e3e5705d6534441d86d4cf401fd725db750cf4946c3b6` |
| E1G0 smoke result | `024cd58e420f72df2322fbc489f1e25ba7d10e2a4106e6c81971fbf12faab35a` |
| E1G1 smoke result | `bc6f2ca9e59554e43b5a3e48112849d808ed2ed663b49000171ee8dffa12a581` |
| Admission summary | `7c06bf4ea765bdbd3f6b5527f612dc6646efefd62713bc7a69141e172cf83d28` |

## C0 midpoint: actual-history integrity and coverage

An independent CPU audit of the immutable first16,384 updates checked131,072
presentations against the pinned sampling and LR plans. Every actual question ID,
full candidate permutation, target association, loss aggregation and finite-value
check passed. Complete accounting equals both the preflight coverage receipt and
the saved step16,384 curve. This verifies the executed prefix, not completion of
the32,768-update run or the remaining four arms.

| Stream | Presentations | Unique questions | Repeated presentations | Available questions |
|---|---:|---:|---:|---:|
| Joint AV |65,536|23,707|41,829|23,707|
| Text rehearsal |16,384|16,384|0|18,432|
| Image rehearsal |16,384|16,384|0|33,696|
| Audio rehearsal |16,384|16,384|0|47,940|
| Video rehearsal |16,384|512|15,872|512|
| Total |131,072|73,371|57,701|—|

The prefix contains24,010 unique observations,11,685 media assets and10,789
source-qualified groups. The independent stream audit verifies uniqueness before
each whole-stream reset: joint AV completed two cycles and Video rehearsal32.
Repeated questions therefore follow the preregistered exhaustion rule, but the
small Video rehearsal pool remains a data-coverage limitation. More presentations
must not be described as more unique training data. No sampling change was made.

At step12,288, C0 first passed all fixed retention gates: joint task-macro
Accuracy0.623087637 and NLL1.060826095; common hierarchical macro NLL1.589783653.
Step16,384 had higher joint Accuracy0.635101809 and lower joint NLL1.016177447,
but failed Audio retention and common-NLL gates (common NLL1.654452839).
The selected best therefore remained12,288. These are development-validation
observations within an ongoing run, not a final result or evidence for gates.

External evidence is `greatest-hits-matched-C0-midpoint-history-audit-01/`:
`audit.py` replays the immutable `history-prefix.jsonl`; `result.json` records
the pinned inputs and complete accounting; `stream-coverage.json` records cycle
checks. SHA-256 values:

| Evidence | SHA-256 |
|---|---|
| History prefix | `53dad695d99cf1e6fd89f00948ead28d3852b4116b495d6df7d2b2d307691ac7` |
| Audit code | `670d49e6c558ce1a772edf18cd402cbaa8bc8ab14e9521c82317c90a6246346f` |
| Audit result | `44eb2fb6a2d30ad3155f3cf80e8ae063b689e94386bf41970790a939b901034b` |
| Stream coverage | `6e02d11367f1aa17aa026bce897bb2fe65fec75631e815285b2d709aa57fa267` |

The selected checkpoint hash was independently checked against `best.json` and
`checkpoint-12288.pt`:
`3b98db3cee66fd5bd6a13f7f11dbfb8732d9b56e8d93af276a2a41215b5d16fd`.
The audit made no optimizer updates, model inference, GPU work or
audit/test/calibration reads. The live source remains pinned to8a5440d and was
not edited; this documentation belongs to the separate evidence branch.

## C0 full update history and fixed endpoint

C0 attempt03 completed all32,768 optimizer updates and262,144 presentations.
The complete actual-history CPU audit and a separate-process replay both pass:
108,187 unique questions,153,957 repeated presentations,25,318 observations,
12,992 media assets and11,369 source-qualified groups. Actual ID order, complete
candidate permutations/targets, LR, finite losses/gradients and coverage match the
pinned plan. The terminal LR is0. This proves the update history; endpoint model
reload and the terminal run receipt are separate requirements.

| Stream | Presentations | Unique questions | Repeated presentations |
|---|---:|---:|---:|
| Joint AV |131,072|23,707|107,365|
| Text rehearsal |32,768|18,432|14,336|
| Image rehearsal |32,768|32,768|0|
| Audio rehearsal |32,768|32,768|0|
| Video rehearsal |32,768|512|32,256|

Every stream exhausts its pool before repetition. Joint AV completes five full
cycles plus a partial sixth; Text one plus a partial second; Video64 complete
cycles. Image and Audio do not complete a full cycle. This matches the prospectively
fixed plan and does not establish sufficient unique Video coverage.

All ten scheduled evaluations were saved. Joint metrics below are equal-task
macro development-validation metrics; common NLL uses the existing hierarchical
macro over four modalities. Eligibility also includes the per-modality2pp
Accuracy retention checks, so a lower common NLL alone does not imply eligibility.

| Update | Joint Accuracy | Joint NLL | Joint Brier | Joint ECE | Common NLL | Eligible |
|---|---:|---:|---:|---:|---:|---|
|0|0.202201|2.350688|0.957725|0.291036|1.630249|yes|
|512|0.441697|1.476249|0.653523|0.056221|1.678375|no|
|1,024|0.434247|1.452946|0.659974|0.086683|1.622150|no|
|2,048|0.433302|1.774114|0.706696|0.192031|1.676056|no|
|4,096|0.489557|1.419100|0.614460|0.061961|1.624888|no|
|8,192|0.530774|1.303677|0.592235|0.092065|1.659337|no|
|12,288|0.623088|1.060826|0.508792|0.074996|1.589784|yes, selected|
|16,384|0.635102|1.016177|0.485637|0.065976|1.654453|no|
|24,576|0.673963|1.008571|0.461208|0.083666|1.735700|no|
|32,768|0.669907|1.026744|0.461959|0.097250|1.776433|no|

The final endpoint improves Joint metrics over the selected checkpoint but fails
the fixed common-NLL retention gate. Selected best remains12,288. Replaying every
selection decision matches the saved curve. Both checkpoint pairs have matching
hashes: `best.pt` equals `checkpoint-12288.pt` with the hash above; `final.pt`
equals `checkpoint-32768.pt` with SHA-256
`924e64a6cb9b7610d35e3845eca94f66b8b5bfe9a1c81af241952aede2f9247c`.

In particular, common Video Accuracy rises from0.3375 to0.3825, while its NLL
worsens from1.501657 to2.766735, Brier from0.755560 to0.931019 and ECE from0.196529
to0.381354. Thus higher Accuracy does not establish improved probability quality.
The other final modality Accuracies are Audio0.629198, Image0.527832 and
Text0.219582. These are scoped development results, far below the product target;
they do not determine a representation-capacity ceiling or a gate benefit.

External full-history evidence is
`greatest-hits-matched-C0-full-history-audit-01/` (`audit.py`, immutable
`history-prefix.jsonl`, `result.json`, `stream-coverage.json`). Audit result SHA-256:
`3255ab2454dd892b852c79c5fea2f60ddcd15526ec3aa5ed4c8e71131839df0c`;
stream result SHA-256:
`0fd37610d406e4791350dc5857e9ff1cbacbecd4494252c40d3fc4b318377442`.
The endpoint/selector observation is
`greatest-hits-matched-C0-step32768-observation-01/result.json`, SHA-256
`c7a6bb09898121c885538c62d19693e42865198e3111102e8d7d551e1ccaef57`.
It explicitly leaves best/final reload and full-run success pending. The live
training source and all legacy runs were unchanged by these CPU observations.

## C0 terminal verification

The pending C0 checks above subsequently completed. Attempt03 ended with
`PASS_MATCHED_AV_TRAIN`, `best_final_reload_verified=true` and no error at
2026-10-17T04:35:32.636966Z (13:35:32 JST). PID1320 exited. Wall time was
12,699.576 seconds (3.528hours), training updates4,337.067 seconds and
evaluation/reloads8,093.611 seconds. No cloud compute was used.

Independent CPU checks verified all91 ledger artifacts and all12 endpoint
score-file hashes against the original scheduled outputs. The production
comparator's single-run validator then recomputed all summaries from all ten
points and six complete score matrices per point; selector replay, endpoint
bindings and both reloads pass. This single-run invocation does not bypass the
CLI's requirement for five completed arms or claim a five-arm comparison.

Terminal training-receipt SHA-256:
`3192d0f3665fbd4558ae3094747af105a1c7441a036151b023e2c2663e94f200`.
Independent receipts and reproducible CPU audit code are under
`greatest-hits-matched-C0-terminal-verification-01/` and
`greatest-hits-matched-C0-saved-metric-audit-01/`. The latter pins analysis
commit a1b8e66767f8819168715506d91c51b57c6f7461. Selected best remains12,288;
final remains32,768. Successful execution does not imply product-quality admission.

E0G0 launch admission was rejected before process/output creation because the
unrelated Orbitune idle trainer automatically occupied the GPU after C0 exited.
Neither a retry during another trainer's initialization nor intervention in that
trainer is authorized by the existing process-protection rule. Permission for its
existing checkpoint-saving STOP/yield mechanism has been requested; no STOP/YIELD,
kill, code change or external checkpoint mutation was performed. E0G0 and the
other three factorial full fits remain unstarted at this observation.

## Remaining requirements

The CPU-only `scripts/summarize_greatest_hits_matched.py` now implements the
post-run saved-prediction comparison. It requires all five complete32,768-update
training receipts with verified best/final reloads, matching source/input/audit
and shared config, exact ordered evaluation metadata and identical step-0 score
matrices. It verifies the complete artifact ledger, including gate/reload arrays,
recomputes every scheduled TRAIN/validation metric from the raw FP32 scores, and
replays the fixed retention/min_delta/earliest-tie rule. A better ineligible
checkpoint cannot become selected. The output retains every curve and reports
fixed equal-step and selected-checkpoint contrasts, resources and code hashes.

The six signed contrasts are E1G1−C0, E0G0−C0, E1G0−E0G0, E1G1−E0G1,
E0G1−E0G0 and E1G1−E1G0, plus E1G1−E1G0−E0G1+E0G0 interaction. Positive
Accuracy deltas and negative NLL/Brier/ECE deltas indicate improvement. Selected
checkpoints can have unequal steps. These descriptive comparisons do not replace
grouped uncertainty analysis, gate-distribution review, input interventions or
the unresolved complementary-task/OOD/product requirements.

Thirty-five CPU tests cover rejection of unfinished/mismatched runs, the
fixed selector, factorial signs, raw-score/padding/metric integrity, required
gate files, and complete synthetic five-arm disk/CLI execution. A self-consistent
but different initial output first escaped detection (RED), then was rejected
after the initial-parity gate was added. The fixture reduces counts only by
test-process monkeypatching; the production CLI retains all six full counts and
all ten fixed evaluation points. Full local checks:2,403 passed,11 skipped with
CUDA hidden; final focused tests:35 passed. Ruff passes. Logs remain under
`greatest-hits-matched-summary-verification-01/` in the external artifact root.
The fixture is implementation evidence, not a real five-arm quality result.

For every factorial arm/view/step, the analyzer also describes the added-residual
audio/video gates: mean, population standard deviation and0/5/50/95/100 percentiles,
both candidate-weighted and with each active question weighted equally. It reports
target/distractor and source-task distributions separately. Valid-candidate gates
must lie in[0,1]; padding and whole-question bypass use the exact-1 sentinel.
Missing-stream views must bypass; joint views must be active. E0G0/E1G0 must use
exactly-one gates when active. Mixed sentinel/range/padding corruption is rejected.
These are descriptive gate diagnostics, not calibrated reliability, global input
importance or evidence of complementary correctness. They never enter selection.

After all five training runs complete, invoke the module with five explicit
`--run ARM=<completed-directory>` arguments and a fresh `--output` directory.
For example, `--run C0=.../greatest-hits-matched-train-C0-03` binds the C0 run;
the other four arguments must bind their own completed arm, not the smoke or
another arm's directory. No aggregate production result has been produced yet.
The analyzer performs no model inference, fitting or fresh-process model reload;
it checks the training-process endpoint reload evidence already saved on disk.

Finish all five fixed 32,768-update curves sequentially, verify selected and final
reloads, and compare fixed equal steps and retention-eligible selected checkpoints.
Retain all failed/ineligible outcomes and the full TRAIN/validation gaps. Follow
with the already-frozen input interventions; replacement sensitivity does not
supply correctness labels for both-required or conflict reasoning. Sparse Video
rehearsal, zero bell TRAIN positives, source/task ambiguity and domain shift remain
explicit limitations. No audit/test/calibration access, cloud cost or product
promotion occurred. Four-modality 95–98% quality, independent OOD/complementarity,
package/latency and the remaining R0–R8 release gates are still unmet.

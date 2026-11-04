# Fresh deterministic control/candidate2048 protocol

Historical nondeterministic runs remain unchanged, including failed long01. This
new pair does not weaken their exact512-prefix gate or claim resume equivalence.

Both use deterministic algorithms=True and cuBLAS workspace:4096:8 before CUDA;
same seed17, encoder-coverage initial checkpoint, data/16384-question order,
validation IDs, AdamW2e-5/wd.01,8questions/update,clip1,dropout off,2048 updates.
Control retains frozen zero selected-row delta; candidate trains343x384 delta.
Both resident counts22,845,313 (control's zero delta is removable). Trainable
10,647,169 versus10,778,881. All6existing blocks+classifier otherwise identical.

Pinned separate16-step traces for each condition must match exact initial weight
hashes and every loss,norm,gradient hash and updated parameter hash through16.
Full initial validation runs before these updates, unlike the trace harness; a
mismatch saves diagnostic details and stops, not a reason to relax the gate.

Control2-step smoke passed best reload/freeze; both conditions'16-step traces
already ran. Start control first, then candidate in separate output, no concurrent
GPU training. Earliest mean original/curriculum validation NLL including0 selects
best, every128 updates, final retained. Compare fixed512/1024/1536/1920/2048 plus
selected checkpoints, grouped paired prediction analysis; no extra seeds/audit.

Disk allowance measured from completed parent output121649559bytes, plus15%
margin per pending condition and512MiB reserve. Current966MB free passes the
~817MB requirement for both runs. No old data/checkpoint deletion. No automatic
extension beyond2048, transfer tuning, calibration, compression or product adoption.

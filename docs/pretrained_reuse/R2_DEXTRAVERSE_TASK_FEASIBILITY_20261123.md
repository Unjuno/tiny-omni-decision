# DextraVerse real-task feasibility follow-up — 2026-11-23

## Decision

Keep DextraVerse **not admitted for an R2 quality run**. The pinned metadata
confirms synchronized real-robot video, tactile maps, force, robot state and
actions, but it does not provide a defensible outcome label for a supplied-
option Decision task that demonstrably requires both video and tactile input.
No archive, episode payload, feature, checkpoint or held-out result was opened
or created for this follow-up.

## Candidate targets reviewed

The source was pinned in the accompanying screen to dataset revision
`257731928e25cd068b02d80d18d8f3cb77075a14`. Its per-task README documents
261 real pick/place episodes and 56 real bottle-cap episodes. The pick task
records camera images, tactile deformation maps, contact force, robot state,
actions and instructions. The cap task records similar streams with a different
tactile SDK/schema. The real-task README does not document object poses, grasp-
stability labels, placement-success labels, or cap-angle/unscrewing outcomes.

Three plausible target formulations were checked against the documented arrays:

1. **Stable-grasp Decision:** no stable/unstable label or object-retention
   measurement is supplied. Deriving the target from later hand-joint commands
   would label the operator/policy's future action, not independently verify a
   stable grasp.
2. **Successful placement or cap progress:** the real-task schema does not
   provide an object trajectory or a task-success field. Inferring success from
   later actions or from the same visual stream being scored would need a new,
   independently validated annotation protocol.
3. **Next-action prediction:** future action arrays do provide a label source,
   but this changes the goal to imitation of the recorded controller/operator.
   It does not establish that tactile and video are both necessary; proprioceptive
   and action-history baselines could explain much of the target.

None of these formulations is suitable for selecting a general multimodal
Decision model without first collecting or validating a separate task label.
The two failed-grasp episodes are not enough to define a robust success target.

## Split and sample limits

The real pick task has 261 episode groups, while the cap task has 56. Frame
windows remain nested within episodes and cannot be counted as independent
groups. The simulated tasks are kept separate because they change the domain,
contact physics and tactile representation. Pooling them would not repair the
real-task group count or establish transfer.

Even if many windows were derived per episode, a windowed corpus would not
create additional independent episodes. A cross-validation design could use
episodes efficiently for an engineering diagnostic after a target is validated,
but it would not create an untouched 200-group audit set while also retaining
substantial independent training and selection data.

## Disposition

- Do not acquire the 44.2 GB pick archive or the 5.8 GB cap archive for a
  quality experiment under the current target definitions.
- Do not turn future action commands into a success label or claim both-input
  necessity from stream availability alone.
- Reopen only after an independently validated real-task label and a frozen
  episode-level evaluation plan exist. A later engineering-only timestamp or
  decoder smoke would not count as Decision-quality evidence.

## Sources

- [Official DextraVerse task page](https://ntumars.github.io/DextraVerse/dataset/)
- [Pinned dataset README](https://huggingface.co/datasets/ntumars/DextraVerse/resolve/257731928e25cd068b02d80d18d8f3cb77075a14/README.md)
- [Pinned real pick task schema](https://huggingface.co/datasets/ntumars/DextraVerse/resolve/257731928e25cd068b02d80d18d8f3cb77075a14/DextraVerse/README_real_robot.md)
- [Pinned real cap task schema](https://huggingface.co/datasets/ntumars/DextraVerse/resolve/257731928e25cd068b02d80d18d8f3cb77075a14/DextraVerse/README_unscrew_bottle_cap.md)

# R2 DextraVerse video/tactile source and task screen — 2026-11-23

## Disposition

`REVIEW_CANDIDATE_NOT_ADMITTED_FOR_QUALITY`. DextraVerse is currently the strongest rights-clear real visual/tactile source found in this screen because its project page and pinned Hugging Face card both declare CC BY 4.0. However, no task-level split or supplied-option Decision target requiring both streams was verified. No task archive, RGB/tactile payload, model, or validation/test sample was downloaded or opened.

## Pinned evidence

- Official project page: <https://ntumars.github.io/DextraVerse/dataset/> (inspected 2026-11-23). Describes 617 episodes and 290,533 frames with RGB, robot state/actions, and fingertip tactile maps; two real tasks plus three simulated tasks; CC BY 4.0.
- Official Hugging Face dataset: <https://huggingface.co/datasets/ntumars/DextraVerse/tree/257731928e25cd068b02d80d18d8f3cb77075a14/DextraVerse>, repo revision `257731928e25cd068b02d80d18d8f3cb77075a14`, public and ungated, card metadata `license: cc-by-4.0`.
- Pinned metadata inventory gives archive lengths: `pick_cube_place_box.zarr.zip` 44,197,744,857 bytes; `unscrew_bottle_cap.zarr.zip` 5,794,218,727 bytes; `handover.zarr.zip` 6,233,893,224 bytes; `pick_and_place.zarr.zip` 4,948,091,395 bytes; `stack_cube.zarr.zip` 3,637,717,786 bytes. No archive was fetched.
- The official project page counts 261 real pick-and-place episodes (138,382 frames) and 56 real cap-unscrewing episodes (16,417 frames). The three simulation tasks have 100 episodes each. These counts are per source task; pooling them would mix real and simulated domains and different action semantics.
- The pinned dataset README says each task has 100 episode boundaries for the simulated tasks and documents 261 real pick episodes. It records two pick episodes that never grasp and three with timestamp stalls. For the cap task it records one 0.24-second stall, a discarded fifth session with unexplained 19–61 N force offsets, and an aborted no-contact attempt.
- The pinned README also warns that video timestamps use a 20 fps clock while physical simulation timestamps use 30 Hz; exact RGB exposure timestamps were not recorded. It reports estimated first-frame camera poses for 300 trajectories, 30 first-frame tactile-pose discrepancies, and 135,704/135,734 frames passing a separate pose-reference check. Those are disclosed data-quality constraints, not a substitute for an independent quality audit.

## Project fit

A plausible real task would be a tactile-assisted manipulation-state Decision, such as estimating whether a grasp is stable from RGB plus fingertip signals. The released arrays contain enough streams to investigate that hypothesis, but the source does not provide an explicit calibrated multi-choice Decision benchmark or ground-truth stable-grasp labels. Tactile-only, image-only, and joint baselines would have to be frozen before any fit, with source-grouped controls for missing, shuffled, and time-offset modalities.

The sample-count constraint is material. The 261-episode pick task is the only real task near the provisional 200-independent-source-group floor. Reserving 200 episodes for a single 200-group evaluation leaves only 61 for all training and checkpoint selection; a nontrivial train/validation/audit partition cannot retain 200 independent groups in the audit while leaving substantial independent training and selection data. The 56-episode cap task cannot meet the floor. Its 16,417 frames and the pick task's 138,382 frames do not create that many independent groups: windows from one episode remain clustered observations. Combining all 617 episodes to inflate counts would mix real and simulated domains and different action semantics, and still would not provide 200 independent real-task groups.

Under the repository's provisional policy, a task is eligible for a `PASS` decision only with at least 1,000 examples and 200 independent source groups. A windowed action-prediction probe could exceed 1,000 rows, but it cannot exceed 261 independent real pick episodes, and its effective group count after a proper train/selection/evaluation split would be lower. Therefore this source can support a bounded engineering diagnostic, but cannot presently support a policy-eligible real-task quality claim. A task-specific target and both-input-necessity evidence are also still absent.

The file sizes also matter: the real pick archive alone is about 44.2 GB compressed, before extraction/cache overhead. Other individual archives are roughly 3.6–6.2 GB. A bounded download should not be planned until a useful target and split are frozen and exact uncompressed/feature-cache sizes are estimated.

## Decision

Keep DextraVerse as the leading **real visual/tactile engineering candidate**, but do not call it an admitted R2 M-2 quality task yet. The next gate is a metadata-only task/split proposal proving that an authentic label requires both video and tactile input, can retain enough independent train/validation groups without pooling unrelated tasks, and has defensible synchronization. If it cannot, retain it only for runtime plumbing and continue searching for an adequately sized task.

No data acquisition, feature extraction, model training, GPU work, or quality claim follows from this screen.

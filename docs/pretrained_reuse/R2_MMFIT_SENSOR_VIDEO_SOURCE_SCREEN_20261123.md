# R2 MM-Fit synchronized sensor/video source screen — 2026-11-23

## Disposition

`REVIEW_NOT_ADMITTED` for the synchronized inertial-sensor + video task. Do not acquire the linked full sensor archive, extract features, or fit a model on this source under the current rights record. This is a metadata-only screen; no MM-Fit payload, annotations, model checkpoint, evaluation data, or sealed audit was accessed.

The separate RGB/depth video record is listed under CC BY 4.0, but that grant does not establish rights for the separate sensor/pose archive. The official MM-Fit project page describes the synchronized wearable streams and links the full archive, but its inspected page shows only a copyright notice and no dataset-specific license. The starter-code repository is MIT; its README says it contains example code and directs users elsewhere to download the dataset. The code license therefore does not clear the data.

## Pinned source evidence

- Official project page: <https://mmfit.github.io/> (inspected 2026-11-23). Describes synchronized RGB-D video, inertial sensors from smartphones, smartwatches and earbuds, pose estimates, and more than 800 minutes of recordings. The page lists device sampling rates and links both the full project archive and a separate RGB/depth Zenodo deposit.
- Official starter repository: <https://github.com/KDMStromback/mm-fit>, default branch `master`, commit `e4b6b7e1fff68e3d93130ed087d23593c9016c94` (GitHub API read 2026-11-23). Repository metadata reports MIT. Its README describes the repository as starter/example code and points users to the project page for dataset download; no data license was found in the inspected official metadata.
- Separate video/depth deposit: <https://zenodo.org/records/7672767>, record `7672767`, DOI `10.5281/zenodo.7672767`, title “MM-Fit Dataset - physical exercise workout sessions (video)”. Zenodo API metadata reports `cc-by-4.0`, open access, and video files. It is explicitly a video/depth part of MM-Fit, not evidence that the separate inertial-sensor archive is covered by that license.
- Official paper: <https://vradu.uk/UbiComp2021.pdf>, Strömbäck et al., IMWUT 2020. It describes exercise recognition and repetition counting, 5-second windows, and session-level splits: 10 workout IDs train, 3 validation, 3 seen-subject test, and 5 unseen-subject test. The paper reports 96.37% multimodal exercise-recognition accuracy and 96.01% for the skeleton-only model on unseen subjects. These are the authors’ results for their own architecture and split, not a result for this repository’s Decision model, not calibrated option probabilities, and not evidence that sensor input is necessary. The numerical difference is only 0.36 percentage points; no paired uncertainty estimate is supplied in the inspected passage.

## Project-fit assessment

The task is relevant in principle: it is temporal, synchronized, and provides both video-derived pose and wearable inertial input. It could test whether sensor streams add information to video on exercise recognition. However:

1. The full sensor/pose archive has no verified dataset license in the inspected official records. The CC BY 4.0 video-only record cannot be extended to that archive by inference.
2. The published validation split has only three workout-session groups, so it does not provide enough independent validation groups for this project’s development gate. The reported hundreds of thousands of windows are highly correlated samples from a small number of sessions, not independent groups.
3. The paper’s fixed exercise classifier is not a supplied-option Decision benchmark and does not establish calibrated probability quality or a necessary contribution from both streams.
4. The published multimodal versus skeleton-only results are close, with no paired uncertainty estimate in the inspected material.
5. The project page describes 20 workout sessions, while the paper’s listed split IDs cover 21 IDs (`0` through `20`; 10 + 3 + 3 + 5). This count discrepancy needs source clarification before any split is adopted.

## Next action

Do not use MM-Fit for the current training path unless the data owner provides explicit terms covering the full sensor/pose archive and the project can freeze a genuinely independent group split. The separate licensed video/depth record could only support a video-only task and would not resolve R2 M-2. Continue searching for a rights-cleared synchronized task with enough independent scene/person/session groups; do not treat this source screen as a quality result.

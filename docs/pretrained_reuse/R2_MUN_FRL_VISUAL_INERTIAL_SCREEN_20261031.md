# MUN-FRL visual-inertial task and source screen — 2026-10-31

## Decision

**Rights are compatible for a future product-directed pilot; the dataset is not
admitted for a quality-bearing R2-M experiment yet.** The current publisher
page explicitly licenses the underlying RGB, LiDAR, GPS/IMU, and calibration
data CC BY 4.0. The linked IJRR article has a different CC BY-NC 4.0 license;
this screen concerns only the dataset files under the separate raw-data grant.
No data payload, media, calibration file, or label was downloaded or opened.

## Pinned source evidence

- Official source repository: `ravindu90/MUN-FRL-VIL-dataset`, `master` HEAD
  observed with `git ls-remote` as `28d5950789461270211ce3759ce00f1ab8edc16d`.
- At that commit, `docs/index.rst` says the raw dataset and calibration files
  are CC BY 4.0, with attribution to the 2024 MUN-FRL paper required. It
  expressly distinguishes the article's CC BY-NC terms from the raw data's
  CC BY grant.
- The source describes two hardware-synchronized RGB cameras at 20 Hz, IMU
  acceleration/angular rate at 400 Hz and magnetic field at 100 Hz, LiDAR at
  10 Hz, and RTK/PPK position ground truth at 5 Hz.
- Nine named ROS-bag sequences are listed, each 27.2–89.9 GB; together they
  total about 435.5 GB. The official page also links 3.55 GB and 6.00 GB
  workshop benchmark bags, but those are not independent sequence groups.
- The paper describes GNSS-denied aerial visual-inertial-LiDAR odometry and
  mapping as the intended task and reports nine sequences / about 100 minutes.

## Candidate task and limits

A decision task could select a short-horizon relative motion option from past
RGB frames plus synchronized IMU; future RTK/PPK displacement would provide
labels only and must never enter the input. The design must compare option and
metadata priors, video-only, IMU-only, and joint models, and include stream
replacement/time-offset controls. Entire flight sequences must stay together
across splits.

Nine sequence groups are too few for a representative independent quality
claim: many windows could be formed, but they would still cluster in only nine
flights, so group-level uncertainty would be based on nine units. The aggregate
payload is also beyond a practical local-first acquisition for this stage; the
small workshop bags can only support a parser/feature smoke, not train/validation
quality. The source page is version-pinned here, but its linked bags have no
published per-file checksums or immutable release revision in the inspected
metadata. Therefore keep this candidate `REVIEW` for R2-M quality and do not
download it or start training from the small workshop subset. No modality or
fusion score is claimed.

## References

- [Official documentation pinned at `28d5950`](https://github.com/ravindu90/MUN-FRL-VIL-dataset/blob/28d5950789461270211ce3759ce00f1ab8edc16d/docs/index.rst)
- [Official MUN-FRL dataset documentation](https://mun-frl-vil-dataset.readthedocs.io/en/latest/)
- [Thalagala et al., MUN-FRL paper](https://doi.org/10.1177/02783649241238358)

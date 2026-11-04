# GrandTour visual-inertial source and task screen — 2026-10-31

## Decision

`REVIEW_NOT_ADMITTED`. GrandTour is the strongest newly identified product-
compatible multi-input candidate in this screen: the official dataset page
explicitly licenses data CC BY-SA 4.0 and describes synchronized camera, IMU,
LiDAR, proprioception and RTK-GPS. However, Hugging Face's dataset card at the
current data revision labels the repository `MIT`, the whole collection is
listed as 2.11 TB, the official dataset page requires registration for ROS
files, and an exact mission-level train/validation plan and typed Decision
task have not been established. No data payload was accessed or downloaded;
no model was trained or evaluated.

## Pinned metadata evidence

- Official website checked 2026-10-31: [GrandTour](https://grandtour.leggedrobotics.com/)
  and its [data overview](https://grandtour.leggedrobotics.com/dataset).
  The source describes 49+ environments, 50,000 steps and 150,000 captured
  images. Sensors include 10 cameras, 8 IMUs, 3 LiDARs, proprioception and
  dual RTK-GPS; reported time synchronization accuracy is 1 ms.
- The official page states dataset data are CC BY-SA 4.0 (including commercial
  use with attribution and share-alike); software is separately MIT. The
  metadata includes `license=mit` in Hugging Face's card at repository SHA
  `daf91c5e2b6db564822a35cc58222d73f8e75bab`, last modified 2026-01-13.
  The data-license conflict is unresolved; follow the official data grant only
  after the host metadata is corrected or the maintainer clarifies in writing.
- Hugging Face reports 2.11 TB total, public and not gated. Its dataset viewer
  cannot parse the archive as a dataset split. The official site says ROS
  access uses registration; it describes ZARR as registration-free via
  Hugging Face. Do not use any official COMFORT hidden test missions or their
  reference trajectories in development.
- Official COMFORT benchmark splits two validation missions (`heap-1.tum`,
  `arc-3.tum`) and six hidden-reference test missions. These are localization
  benchmark partitions, not a ready-made split for a learned Decision head.
  Every mission is one continuous deployment, so split and uncertainty units
  must be whole missions, not windows.

## Candidate task and limitations

A task-matched pilot could ask a small Decision head to choose among fixed
short-horizon relative-motion bins using past camera frames plus synchronized
IMU; future RTK/PPK displacement could generate labels but must never enter
the input. Compare fixed-option/metadata priors, camera-only, IMU-only, joint,
and time-shifted/replaced-stream controls. Report the target as a robotics
navigation decision, not general-purpose multimodal reasoning. The existing
COMFORT trajectory benchmark is continuous pose estimation, so it cannot be
relabelled as a Decision benchmark without a frozen task protocol and
independent mission-level groups.

Before admission, resolve the data-license metadata mismatch, inventory
mission-level ZARR sizes and hashes without opening hidden test labels,
establish enough non-test missions for grouped train/validation, verify sensor
availability and time alignment, and estimate local disk use before any
download. The 2.11 TB aggregate is incompatible with wholesale local
acquisition; selective mission access may still make a small, bounded pilot
possible. Do not fetch payloads or start a run until those gates pass.

## Reproducibility and disposition

Metadata-only review on 2026-10-31. Hugging Face API revision and card fields
are captured in
[`R2_GRANDTOUR_SOURCE_SCREEN_20261031.json`](R2_GRANDTOUR_SOURCE_SCREEN_20261031.json).
Official dataset source and benchmark pages were read; the GitHub examples
repository HEAD was observed as `ef8df8ab3a9452087ee02e0016a58f8a3db6da49`.
No external repo was cloned, no data/labels/test annotations were opened, no
files were persisted from the source, and no GPU was used.

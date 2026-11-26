# RoboMNIST cross-task metadata exposure — 2026-11-03

## Finding

The two validation groups missing from the current AV-match feature cache also
appear as audio candidate source references in the TRAIN manifest for the
separate `robomnist-av-correspondence-seed17-attempt-05` task:

- `Ar1_Vhigh_Ac1_Rp10_UNC085`: four TRAIN rows reference this source group.
- `Ar1_Vhigh_Ac1_Rp22_UNC304`: four TRAIN rows reference this source group.

The TRAIN manifest SHA-256 is
`0aa00362dc933c85926cd11bde8774ce96d1d350468e5927f226f04628b96049`.
The attempt's `summary.json` SHA-256 is
`538d526bfa049d4cef29568bcbe06b32a96a303a2c229d10cf0420641932d0cb`.

## Interpretation

This is **metadata-manifest overlap**, not evidence that either group was used
to train or evaluate a model in the earlier attempt. Its summary explicitly
records `payload_opened=false`, `training=false`, `model_inference=false`, and
`sealed_audit_evaluated=false`. No media, feature payload, validation manifest,
or sealed-audit manifest was opened in this follow-up.

The two groups remain part of the current RoboMNIST fixed validation split;
their within-task split assignment is unchanged. However, they are not
project-wide unseen metadata identities. Any later result must describe this
cross-task metadata exposure and must not call these records a newly blind
project-level audit. This does not establish actual model leakage or alter the
current incomplete feature cache.

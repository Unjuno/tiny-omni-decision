# ADL: exact split-content identity for two activities

The [fixed protocol](R2_ADL_TWO_ACTIVITY_CONTENT_PROTOCOL_20261027.md) compared
sensor 1 of the two fully publisher-MD5-verified activities under the fixed
participant split. It read 583 training and 154 development-validation rows.
The complete int16 audio arrays and each of the three vibration axes were
hashed independently with their lengths. For **each channel**, all 583 training
and 154 validation hashes were unique and no hash crossed the split. No empty or
constant channel was found. Raw waveform and per-row hash data were not retained.

This is an exact-byte check, not a session-disjointness, partial-clip overlap,
physical synchronization or similarity guarantee. It covers only sensor 1 and
two activities, and no model was fit or scored. The source is not admitted for
training based on this result.

Receipt:
`C:/CodexArtifacts/pretrained-reuse-r2/adl-two-activity-content-20261027-attempt-01/result.json`,
SHA-256 `8ed5064d24ad3e9e6f135e41e014dbb38d55bf825f2f42ccb3276294884f804c`.
The protocol, source archive, split, and executed script hashes are in the
receipt. Three targeted tests and Ruff passed.

The concurrent 25-activity partial inventory separately reported `24_wipe-stove`
as `UNSUPPORTED_SIZE`: its selected first member declares 135,847,664 decoded
bytes, exceeding the preregistered 128 MiB limit. It was not fetched or
inspected. This is a missing feasibility result, not a missing or failed data
class. The inventory continues with the original bounds.

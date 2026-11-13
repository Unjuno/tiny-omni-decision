# R2-M source and interface readiness

Read-only audit performed while the prior token experiment ran; no live code
was changed. Remote R2-M commits `c098ace`/`afdfe7c` were read before and merged
after that run terminated successfully. The prior optional-text probe family is
now closed for this bounded comparison; native/no-text and sensor-inclusive work
is the next priority. This inventory does not complete M-0 or M-1.

## Existing controls and API gaps

STARSS23 candidate_inputs uses cached observations, one-hot event IDs and typed
binary option vectors. It has no MiniLM question input in that path, so it is a
reusable no-text, typed-event control. Its imbalanced narrow tasks and negative
fusion result do not establish the new task matrix. It lacks explicit missing
versus zero-signal handling. Current cache keys also lack screen modality; current
DecisionOutput JSON requires a question name. A versioned ObservationBatch with
true token timestamps, valid masks, physical units and calibration is still needed.

## Sensor source findings

UCI341's downloaded archive README explicitly prohibits commercial use, whereas
the [official catalog](https://archive.ics.uci.edu/dataset/341/smartphone%2Bbased%2Brecognition%2Bof%2Bhuman%2Bactivities%2Band%2Bpostur)
states CC BY4.0. Record the conflict as REVIEW; do not assume which governs.
Archive/README hashes are saved. No raw sensor signals, labels or test contents
were opened in this audit. This supplements the earlier MOMENT audit; it does
not change the historical embedding-smoke result.

HARTH's [official UCI release](https://archive.ics.uci.edu/dataset/779/harth)
is the22-subject version with explicit CC BY4.0. Its310,795,012-byte archive was
retrieved and pinned SHA256
`2ab54d6b11467ddbe998dfb60355f9f1ae181b0af7a89e7c78bb7af2cf16b64a`.
It contains CSV files only; no contradictory bundled license was found in the
member inventory. Admit this official release for bounded sensor research with
attribution. The author's31-subject v2 repository was separately pinned to
`dad2cfbe89a26f72f19770419469ac037de200df`, but is not substituted for the UCI
release; its metadata-only audit retains an unresolved dataset-license scope.

Before reading any CSV, sorted member IDs were shuffled with Python Random17:
14 train,2 calibration,3 validation,3 reserved sealed-audit subjects. Splits and
hash are saved. Only the first128 rows of training subjectS025 were inspected
for format after verifying its full ZIP member CRC and SHA. All six accelerometer
channels are finite; timestamp gaps range0.02–1.54s. Thus raw data must not be
blindly reshaped into continuous50Hz windows. Windowing must split at gaps and
record time/validity; official units are g. No calibration/validation/audit CSV
was opened and no model was trained. This source does not establish paired
video/audio availability. Real M3-AVS remains NOT TESTABLE from these assets.

Artifacts: `C:/CodexArtifacts/pretrained-reuse-r2/r2m-source-readiness-01`.
[Hashes, partitions, provenance and format result](R2M_SOURCE_READINESS_20261012.json).
Next: implement/test the no-text observation contract and gap-aware sensor
preprocessing; select screen and genuine multi-signal sources before claiming
M-0 complete. Synthetic fusion smoke may verify wiring only.

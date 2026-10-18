# RoboMNIST complete directory-only admission protocol

Purpose: establish complete physical-recording inventory for a public synchronized
audio/video/optional-CSI Decision candidate. Full <=200M prequantization native
text/image/audio/video calibrated JSON and95–98% joint/OOD goal remains unmet.
User authorizes public-data selection and autonomous bounded local work.

Extend the existing source auditor directly; no new architecture/learning.
Read official version1 metadata already pinned SHA256
`18ef8b20c82056747c58572d24f262b2d12c9284e4ffd83c245ae64eca2ab120`.
Dataset CC BY4.0 grant and all60 class archive IDs/sizes are rechecked.
All60 directories, ordered class00–59, not a quality-selected prefix.

Reuse exact Content-Range/HTTP206/length/TLS checks in existing
`scripts/fetch_starss23_smoke.py:HttpRangeReader`; only ZipFile.infolist(), never
open/read/extract a member. Each archive512KiB transfer maximum; overall12MiB
including already charged failed/short responses from the reader. Sequential
network execution, body900s with checks before/after each read, no retry/fallback.
Network timeout may finish after the body deadline; record actual overrun as
incomplete, never as admitted success. Metadata output cap16MiB,1GiB free-disk
reserve applies to this small CPU/network task only. Existing GPU diagnosis's
8GiB reserve is unchanged and its failure remains protected; no GPU job started.

Validate every real member root against exact inspected archive stem; original
class/file labels, types, path/duplicate checks unchanged. Preserve every genuine
repetition/uncertainty group; all3 camera +3 mic +1 CSI streams together. Position
JSON supervision-only. Resource-fork exclusion is inventory cleanup, not dropping
difficult observations. Record counts per archive and repetition, unknowns and
missing source computed-MD5. Full archive MD5/CRC of member payloads not verified.

Fresh output directory, config/protocol/code/source hashes, PID/start/end/command,
range-byte totals and per-archive progress. Terminal incomplete record on faults;
no automatic restart or reuse of unverified prefix as a complete admission.
Independent CPU re-audit of all saved directories and groups after terminal
success. All previously captured provider/legacy/model artifacts read-only.

PASS means60-directory complete inventory/grouping only, not content/sync/split,
training eligibility, taskquality or fusion benefit. No media/GT/payloads, training,
model inference, official validation/audit scoring, cloud, quantization, merge or
product adoption. Future split must group these physical records and also account
for unresolved collection-session/near-duplicate content identities; one lab/two
robots does not establish population OOD.

Implementation uses TDD, inline execution and one final reviewer. Public source
selection approval is already provided; no repeated design permission requested.

# Fixed development media acquisition and exhaustive integrity decode

After the full733-parent TRAIN annotation inventory, keep the prospective308 TRAIN /
101 development-validation partition and324 quarantined parents unchanged. All17
material,2 action and5 reaction values in validation are supported by TRAIN; do not
balance/reassign dates to reduce the measured distribution shift. Missing labels
stay missing and all rows remain accounted for.

Acquire `_denoised.mp4` for **all409** development parents, never quarantined or
official TEST media. Pinned directory predicts632,267,494 compressed bytes and
643,096,543 extracted bytes; largest member32,950,656 bytes fits the existing32MiB
member cap. Use2 bounded independent workers, total worst-case transfer<=1GiB,
dispatch deadline1800s, HTTPS/IPv4/If-Match, exact206/range/ETag/CRC/SHA. For these
larger members explicitly use90s request/95s subprocess caps; the previous25s
annotation default remains unchanged. Keep all raw range receipts and failures.
No automatic retry, no full archive download/hash claim, no existing output reuse.

Freeze parent/member list and source/partition hashes before requests. Then decode
every acquired video and audio stream on CPU only: record dimensions/rates/frame
and sample counts, finite PCM, monotonic PTS, decoded timing, full decoded content
hashes and duplicate identities. Check every annotation onset and the prospective
0.5s event window[t-0.25,t+0.25) against both stream extents; retain each failing row
with reason. The observed adjacent onsets are>0.25s apart, so the centered window
does not include another annotated onset; this is not proof that there are no
unannotated sounds or that long scratches are temporally isolated.

Native tasks remain material17-way, action2-way and reaction5-way; keep every
known field, including partial rows. Event timing is supplied as localization
metadata, not predicted by the Decision model. This probe does not measure event
localization, semantic label accuracy, physical AV offset or useful fusion. Task
construction and model training require the actual integrity result. Coarse date
groups and unknown repeated sites/actors remain explicit limitations.

CPU tests precede new code execution. Save complete local evidence, then publish
small results/config/code/tests with Ruff/pytest/manifests and exact-commit CI.
No model/encoder load, training, calibration, sealed audit, cloud or product claim.

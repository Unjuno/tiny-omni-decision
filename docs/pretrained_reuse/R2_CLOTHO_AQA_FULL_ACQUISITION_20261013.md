# Full frozen AQA development acquisition: started

Source commit060c7ab, command
`python -m scripts.fetch_clotho_development_media`.
Output: `C:/CodexArtifacts/pretrained-reuse-r2/clotho-aqa-development-media-01/`.
The process was confirmed live as PID69416; the first verified progress was
16/856 files. This launch record does not claim completion. The atomic local
result.json is authoritative for current progress and terminal PASS/FAIL.

All three frozen split SHA-256 values are checked before acquisition. Four
independent HTTP range readers each enforce512MiB total transfer, exact206 and
Content-Range, TLS, and bounded retries. Only selected member payloads are read.
CRC, uncompressed length, per-file SHA, license and16MiB maximum member size
are checked. Admission requires5GiB initial free space and2GiB runtime reserve.
Existing16-clip probe and the reviewed source16854 member are reused only after
hash and ZIP CRC checks. The explicit alias additionally requires its pinned SHA.

No split/question/label is replaced or removed, and no media/checkpoint enters
Git. Official AQA validation/test question files and sealed audit remain unopened.
This is frozen development media acquisition, not model training. Full waveform
integrity, feature extraction and audio-only/joint training remain pending.
The script passed Ruff before execution. Do not restart solely because observing
the live process times out; inspect the same process/session and artifact state.
If acquisition fails, preserve partial output and diagnose before any new attempt.

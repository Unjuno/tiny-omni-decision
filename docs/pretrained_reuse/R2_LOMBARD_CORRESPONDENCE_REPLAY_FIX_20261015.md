# JSON-domain replay comparison correction

The immutable6fffb73 producer completed primary01, but replay01 failed at report
comparison. `build_quartets` returns integer Hamming-distance count keys; JSON
deserializes them as strings. Persisted primary/replay reports have identical
catalog/model/temperature/calibration/policy values. This is a serialization
comparison defect, not evidence of completed replay gates: later array/terminal
checks were not reached in the failed attempt.

Preserve all original source, artifacts and the failed receipt. A separate v2
producer copies the original execution path with only JSON-domain finite report
equality and explicit binding of this correction/code/test. No float rounding,
tolerance, data/model/scoring/calibration/resource change. NaN fails closed.
Fresh primary02 and replay02 use new directories; complete all original exact
NPZ/NPY/JSON and terminal gates. No re-fitting or metric-based experiment change.
The repeated inference is for validation of software, not quality selection.

# Frozen source-review sample clock preflight

On the 16 already frozen official-train videos, verify each original MP4 SHA
before reading it. Decode audio timestamps and sample counts and use the existing
exact `sample_interval` contract to determine whether the annotated half-open
microsecond event interval is covered without decoded gaps/overlaps. Decode
video from a backward seek to find the first timestamp inside the same interval;
record the actual frame PTS and its time base. Preserve every sample including
failures and reasons; do not change times to make them pass. This is a container
clock/coverage preflight, not proof of physical audio-video synchronization,
audibility, annotation correctness or causal source identity.

Read only the frozen train sample and its pinned MP4s; do not inspect model
predictions, validation or sealed audits. No training, media copying, further
acquisition, checkpoint selection or mutation. Save a new small result receipt
with sample, script and individual media hashes, explicit library version and
all per-record outcomes. Follow-up human review must inspect complete synchronized
video plus audio; timestamps or still images do not substitute for it.

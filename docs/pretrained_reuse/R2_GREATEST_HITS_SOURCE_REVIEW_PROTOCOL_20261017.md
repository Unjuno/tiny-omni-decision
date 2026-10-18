# Fixed TRAIN source review before task admission

Review only the existing development TRAIN media. Keep every frozen split,
label, window, model and prior result unchanged. No training, validation-media
inspection, model scores or automatic exclusion/relabeling.

Before viewing new media, choose one parent per each of the21 TRAIN dates: the
minimum SHA256(`17:`+parent) within that date. Within that fixed parent choose the
earliest event with all three released labels, or row0 if none is complete. Keep
missing/ambiguous cases visible; do not replace them. This is a date-coverage
inspection of labeled events, not a representative estimate of label accuracy.

Read only those selected TRAIN annotation members. Verify the pinned partition,
member/raw-media hashes and prior complete-decode receipt. Store the exact selected
IDs/rows before media decoding. For each, preserve the native0.5s mono96kHz window
and every actual video frame in that same interval. Save PTS, a4-column contact
sheet, raw RGB and PCM arrays,10ms RMS bins and strongest-bin offset from the
released onset. A centered amplitude peak is not a measurement of AV offset.
No denoising, gain change, crop or retiming. Decode in the existing PyAV CPU venv.

Inspect all21 contact sheets for visible action, material ambiguity, reaction,
camera/scene cues and whether evidence would be lost by the old central crop.
Record qualitative findings per sample with uncertainty. Do not infer identities
of people or a proven absent site overlap from visual similarity/difference.
Do not report contact timing more precisely than the actual frames permit.

Read the author's project page and paper/appendix for annotation provenance and
timing limitations. Only an author-released parser can establish an official file
schema; unrelated reproductions do not supply missing source guarantees. Distinguish
native material/action/reaction prediction from a both-required multimodal task.

End with an explicit source/task disposition and one executable next action.
If the source does not justify complementary-task truth labels, do not quietly
promote it to that benchmark. It can remain a narrower development source with
documented limitations; the product objective and its evaluation gates stay intact.

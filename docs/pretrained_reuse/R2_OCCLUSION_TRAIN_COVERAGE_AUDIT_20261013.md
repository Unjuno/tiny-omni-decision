# Available training coverage beyond the paired diagnostic

The pinned train annotation archive contains2,184 videos,116 marked cup-game
videos, and99 videos in the current fixed pair corpus. The adjacent JSON lists
the17 remaining cup-game IDs and their original location-question option sets.
Examples have different initial/final option sets (including a fourth object's
position), so they cannot simply be inserted into the existing shared-options
pair representation. Preserve each question's supplied options and answer text;
do not fabricate matching distractors or change the existing benchmark.

The existing99 videos contain780 multiple-choice questions across10 exact text
templates, whereas the fixed probe uses198. Available questions include count,
identity, transparency, handles, camera motion and whether the hidden object moved.
These are possible grounding supervision, not automatically clean temporal labels
or independent examples. All questions from each video must remain in its split.
Their answer priors and cut-frame compatibility need checking before training.

This is a source-coverage inventory, not evidence that more supervision improves
the model. The17 additional videos are not yet admitted/downloaded/extracted by
this audit. No new training or validation evaluation was run. A separately versioned
question-specific candidate representation would support the real custom-option
product better than extending a shared-option-only diagnostic by filtering data.
Legacy paired corpora, results and checkpoints remain unchanged.

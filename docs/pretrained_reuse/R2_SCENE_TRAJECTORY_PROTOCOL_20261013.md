# Scene-conditioned trajectory readout — fixed protocol

Hypothesis: pooled frozen scene features may add viewpoint/context information
missing from the initial-point descriptor and trajectory channels. This is not
an assertion that VideoMAEv2 encodes person-relative orientation.

Use the admitted existing356 video caches and unchanged198 train /514 development
questions. Encoders remain frozen; no annotation IDs/boxes enter inputs. Retain
the documented33 one-frame endpoint differences. Do not reextract old caches.

For each clip, average the existing VideoMAEv2 window vectors (384 dimensions),
project to64 values and add to the candidate query before trajectory attention.
Use the original512-channel trajectories. The scene projection has24,576 weights;
total head188,608 parameters. Resident component count including TAPIR, VideoMAEv2
and MiniLM is70,804,045; this excludes other Omni modalities and is not a complete
product size. No larger pretrained model or encoder update.

Keep seed17,50 epochs,AdamW0.001/weight_decay0.01,batch8 video pairs, earliest minimum
validation NLL, fresh head initialization. Preserve untrained/best/final weights,
all histories and predictions. No seed/policy search after viewing results.
Compare the fixed trajectory-only run with this run, explicitly noting additional
parameters and scene preprocessing. This is not an isolated parameter-count test.

Controls at selected weights: normal, all-zero, paired-video mismatch, scene-only
mismatch and trajectory-only mismatch. Report each modality-equivalent feature
block's removal as a diagnostic, not as a separately trained baseline. Report
Accuracy/NLL/Brier/ECE, both-correct and changed-target pairs, overfitting, resource
cost and exact reload. Video-cluster paired intervals use10,000 draws,seed17.
The current reused development set is not blind. Reject an aggregate-only gain
without evidence sensitivity and condition-dependent correct answers. No sealed
or product promotion. If not useful, close this comparison without a capacity sweep.

Implementation stage: head unit tests pass; full input admission/training runner
and real comparison are pending. No run has started under this protocol yet.

# Native speech audio-dependence control

The completed matched-speech result is99.9484% for both fixed conditions.
Before interpreting this as audio evidence, run the same frozen scorer once with
the audio row at(i+969)%1938. Keep target/options/order/question unchanged.
Assert every pair has different speaker identity; fail if not. Do not search
permutations by score or exclude coincidental content matches. This is a bijection,
so all original assets are used once. Preserve matched-speech outputs unchanged.

Same FP32 model, mean transcript/EOS logprob, temperature1, all1,938rows and
random/hard choices. New native-speech-rotated-01 output; log actual source audio
ID/speaker/hash alongside original query ID. No training or scoring changes.
Report all metrics and matched-versus-rotated Accuracy. This is post-hoc evidence
about audio dependence, not an independent blind accuracy claim. Speaker-cluster
intervals on the earlier hard-minus-random matched comparison do not represent
uncertainty in broad recognition or this two-speaker-dependent rotation.

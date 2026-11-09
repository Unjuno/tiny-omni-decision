import assert from 'node:assert/strict'
import test from 'node:test'

import {
  assertNoCrossSplitAudioOverlap,
  pcmWavInfo,
} from '../scripts/materialize_speech_commands_fresh_m1.mjs'

function pcm16MonoWav() {
  const bytes = Buffer.alloc(44)
  bytes.write('RIFF', 0, 'ascii')
  bytes.writeUInt32LE(36, 4)
  bytes.write('WAVE', 8, 'ascii')
  bytes.write('fmt ', 12, 'ascii')
  bytes.writeUInt32LE(16, 16)
  bytes.writeUInt16LE(1, 20)
  bytes.writeUInt16LE(1, 22)
  bytes.writeUInt32LE(16_000, 24)
  bytes.writeUInt32LE(32_000, 28)
  bytes.writeUInt16LE(2, 32)
  bytes.writeUInt16LE(16, 34)
  bytes.write('data', 36, 'ascii')
  bytes.writeUInt32LE(0, 40)
  return bytes
}

test('fresh M1 materializer accepts only mono PCM16 16 kHz WAV headers', () => {
  assert.deepEqual(pcmWavInfo(pcm16MonoWav()), {
    channels: 1,
    sample_rate: 16_000,
    bits_per_sample: 16,
    bytes: 44,
  })
  const stereo = pcm16MonoWav()
  stereo.writeUInt16LE(2, 22)
  assert.throws(() => pcmWavInfo(stereo), /mono PCM16/)
  assert.throws(() => pcmWavInfo(Buffer.alloc(44)), /not a WAV/)
})

test('fresh M1 materializer rejects exact audio-content overlap across splits', () => {
  assert.doesNotThrow(() => assertNoCrossSplitAudioOverlap([
    { split: 'train', audio_sha256: 'a' },
    { split: 'validation', audio_sha256: 'b' },
    { split: 'train', audio_sha256: 'a' },
  ]))
  assert.throws(() => assertNoCrossSplitAudioOverlap([
    { split: 'train', audio_sha256: 'same-content' },
    { split: 'validation', audio_sha256: 'same-content' },
  ]), /crosses splits/)
})

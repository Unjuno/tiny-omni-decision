import assert from 'node:assert/strict'
import test from 'node:test'

import { decodeTextOrWav } from '../scripts/fetch_speech_commands_probe.mjs'

test('Speech Commands parser preserves WAV byte arrays while decoding text fields', () => {
  const wav = Buffer.alloc(48)
  wav.write('RIFF', 0, 'ascii')
  wav.write('WAVE', 8, 'ascii')
  const encodedText = new TextEncoder().encode('speaker-17')

  assert.equal(decodeTextOrWav(encodedText), 'speaker-17')
  assert.deepEqual(decodeTextOrWav(wav), wav)
})

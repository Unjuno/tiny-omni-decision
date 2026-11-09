import assert from 'node:assert/strict'
import test from 'node:test'

import { chooseGroupRows, LABELS } from '../scripts/preflight_speech_commands_fresh_m1.mjs'

function rowsForSpeakerCount(count) {
  const rows = []
  for (let speaker = 0; speaker < count; speaker += 1) {
    for (let label = 0; label < LABELS.length; label += 1) {
      rows.push({
        source_file: 'train-00000-of-00006.parquet',
        source_row: speaker * LABELS.length + label,
        label,
        target: LABELS[label],
        speaker_group_sha256: `speaker-${speaker}`,
      })
    }
  }
  return rows
}

test('fresh M1 preflight assigns deterministic, label-balanced disjoint speaker cohorts', () => {
  const first = chooseGroupRows(rowsForSpeakerCount(600))
  const second = chooseGroupRows(rowsForSpeakerCount(600))
  assert.deepEqual(first.train.map((row) => row.id), second.train.map((row) => row.id))
  assert.equal(first.train.length, 1500)
  assert.equal(first.validation.length, 1000)
  assert.equal(first.validationGroups.size, 200)
  assert.equal(first.trainGroups.size >= 100, true)
  assert.equal(first.calibrationGroups.size, Math.ceil(first.trainGroups.size * 0.2))
  assert.equal(first.fitRows.length + first.calibrationRows.length, first.train.length)
  assert.equal(first.train.some((row) => first.validationGroups.has(row.speaker_group_sha256)), false)
})

test('fresh M1 preflight fails closed when training group capacity cannot meet quotas', () => {
  assert.throws(() => chooseGroupRows(rowsForSpeakerCount(400)), /distinct speakers|speaker cap/)
})

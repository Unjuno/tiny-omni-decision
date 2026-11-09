import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
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

test('fresh M1 training permits repeated class rows within the frozen five-row speaker cap', () => {
  const rows = rowsForSpeakerCount(600)
  for (let speaker = 200; speaker < 345; speaker += 1) {
    rows.push({
      source_file: 'train-00001-of-00006.parquet',
      source_row: 100_000 + speaker,
      label: 7,
      target: LABELS[7],
      speaker_group_sha256: `speaker-${speaker}`,
    })
  }
  const result = chooseGroupRows(rows)
  const offRows = result.train.filter((row) => row.label === 7)
  assert.equal(offRows.length, 150)
  assert.equal(new Set(offRows.map((row) => row.speaker_group_sha256)).size < offRows.length, true)
  const perSpeaker = new Map()
  for (const row of result.train) {
    perSpeaker.set(row.speaker_group_sha256, (perSpeaker.get(row.speaker_group_sha256) ?? 0) + 1)
  }
  assert.equal(Math.max(...perSpeaker.values()) <= 5, true)
})

test('development speaker selection preserves rare-class coverage within the fixed group budget', () => {
  const base = rowsForSpeakerCount(600)
  const groups = [...new Set(base.map((row) => row.speaker_group_sha256))].sort((left, right) => {
    const hash = (value) => createHash('sha256').update(value).digest('hex')
    return hash(`r2-speechcmd-m1-seed17|${left}`).localeCompare(
      hash(`r2-speechcmd-m1-seed17|${right}`),
    )
  })
  const leftGroups = new Set(groups.slice(200, 400))
  const rows = base.filter((row) => row.label !== 5 || leftGroups.has(row.speaker_group_sha256))
  for (const [index, group] of [...leftGroups].entries()) {
    rows.push({
      source_file: 'train-00001-of-00006.parquet',
      source_row: 200_000 + index,
      label: 5,
      target: LABELS[5],
      speaker_group_sha256: group,
    })
  }
  const result = chooseGroupRows(rows)
  const fixedHashFirst200 = new Set(groups.slice(0, 200))
  const first200LeftSupport = new Set(rows
    .filter((row) => row.label === 5 && fixedHashFirst200.has(row.speaker_group_sha256))
    .map((row) => row.speaker_group_sha256)).size
  const selectedLeftSupport = new Set(result.validation
    .filter((row) => row.label === 5)
    .map((row) => row.speaker_group_sha256)).size
  assert.equal(first200LeftSupport, 0)
  assert.equal(selectedLeftSupport, 100)
})

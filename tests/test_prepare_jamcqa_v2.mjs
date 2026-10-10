import test from 'node:test'
import assert from 'node:assert/strict'

import { normalizeRows } from '../scripts/prepare_jamcqa_v2_dev.mjs'

const categories = [
  ['culture', 4],
  ['custom', 4],
  ['regional_identity', 4],
  ['geography', 4],
  ['history', 4],
  ['government', 4],
  ['law', 4],
  ['healthcare', 4],
  ['jsdf', 20],
]

function rowsWithCategoryCounts() {
  const rows = []
  for (const [category, count] of categories) {
    for (let index = 0; index < count; index++) {
      rows.push({
        qid: `jamcqa-dev-${String(rows.length).padStart(5, '0')}`,
        category,
        question: '日本語の質問',
        choice0: '選択肢A',
        choice1: '選択肢B',
        choice2: '選択肢C',
        choice3: '選択肢D',
        answer_index: 2,
      })
    }
  }
  return rows
}

test('normalizes the pinned dev shape and retains duplicate options', async () => {
  const rows = rowsWithCategoryCounts()
  rows[0].choice1 = rows[0].choice0

  const { normalized, categoryCounts } = await normalizeRows(rows)

  assert.equal(normalized.length, 52)
  assert.equal(normalized[0].split, 'dev')
  assert.deepEqual(normalized[0].options, ['選択肢A', '選択肢A', '選択肢C', '選択肢D'])
  assert.equal(normalized[0].target, '選択肢C')
  assert.deepEqual(categoryCounts, Object.fromEntries(categories))
})

test('rejects duplicate question ids', async () => {
  const rows = rowsWithCategoryCounts()
  rows[1].qid = rows[0].qid

  await assert.rejects(normalizeRows(rows), /invalid or duplicate dev question id/)
})

test('rejects an unexpected category distribution', async () => {
  const rows = rowsWithCategoryCounts()
  rows[0].category = 'unexpected'

  await assert.rejects(normalizeRows(rows), /unexpected category mix/)
})

import type { GigaChatRuleEvaluationResponse, GigaChatRuleEvaluationRow, GigaChatRulePack, GigaChatRulePackFilter } from './types'

const RULE_FILTER_FIELD_ALIASES: Record<string, string[]> = {
  'Трайб': ['Во. Группа'],
  'драйвер': ['Во. Тематика'],
}

function splitStringList(raw: unknown) {
  if (Array.isArray(raw)) {
    return raw.map((item) => String(item ?? '').trim()).filter(Boolean)
  }
  return String(raw ?? '')
    .replace(/\r/g, '\n')
    .split(/\n|,/u)
    .map((item) => item.trim())
    .filter(Boolean)
}

function splitKeywordList(raw: unknown) {
  if (Array.isArray(raw)) {
    return raw.map((item) => String(item ?? '')).filter((item) => item.trim())
  }
  const text = String(raw ?? '').replace(/\r/g, '\n')
  if (text.includes('\n') || text.includes(',')) {
    return text
      .split(/\n|,/u)
      .map((item) => item.trim())
      .filter(Boolean)
  }
  return text.trim() ? [text] : []
}

export function parseRulePacks(raw: unknown): GigaChatRulePack[] {
  const text = String(raw ?? '').trim()
  if (!text) return []
  try {
    const parsed = JSON.parse(text) as unknown
    if (!Array.isArray(parsed)) return []
    return parsed.flatMap((item, index) => {
      if (!item || typeof item !== 'object' || Array.isArray(item)) return []
      const record = item as Record<string, unknown>
      const filters = Array.isArray(record.filters)
        ? record.filters.flatMap((filterItem) => {
          if (!filterItem || typeof filterItem !== 'object' || Array.isArray(filterItem)) return []
          const filterRecord = filterItem as Record<string, unknown>
          const field = String(filterRecord.field ?? '').trim()
          const op = String(filterRecord.op ?? 'eq').trim().toLowerCase() === 'ne' ? 'ne' : 'eq'
          const value = String(filterRecord.value ?? '').trim()
          return field && value ? [{ field, op, value } satisfies GigaChatRulePackFilter] : []
        })
        : []
      const type = String(record.type ?? 'assign_tag').trim() === 'reclass_topic' ? 'reclass_topic' : 'assign_tag'
      return [{
        code: String(record.code ?? `RULE_${index + 1}`).trim() || `RULE_${index + 1}`,
        description: String(record.description ?? '').trim(),
        enabled: Boolean(record.enabled ?? true),
        type,
        source_fields: splitStringList(record.source_fields),
        keywords: splitKeywordList(record.keywords),
        filters,
        target_tag: String(record.target_tag ?? '').trim() || null,
        target_topic: String(record.target_topic ?? '').trim() || null,
      }]
    })
  } catch {
    return []
  }
}

function normalizeMatchText(value: unknown, { strip = true }: { strip?: boolean } = {}) {
  const text = String(value ?? '')
  return (strip ? text.trim() : text).toLowerCase().replace(/\s+/gu, ' ')
}

function escapeRegExp(value: string) {
  return value.replace(/[.*+?^${}()|[\]\\]/gu, '\\$&')
}

const MATCH_WORD_CHARS = '0-9A-Za-zА-Яа-яЁё'

function isMatchWordChar(value: string) {
  return /^[0-9A-Za-zА-Яа-яЁё]$/u.test(value)
}

function keywordMatchesText(keyword: string, text: unknown) {
  const normalizedKeyword = normalizeMatchText(keyword, { strip: false })
  const normalizedText = normalizeMatchText(text, { strip: false })
  if (!normalizedKeyword.trim() || !normalizedText) return false
  if (!normalizedKeyword.replace(/\*/gu, '').trim()) return false
  const pattern = normalizedKeyword
    .split('*')
    .map(escapeRegExp)
    .join('.*')
    .replace(/\\ /gu, '\\s+')
  try {
    const leftBoundary = !normalizedKeyword.startsWith('*') && isMatchWordChar(normalizedKeyword[0])
      ? `(?:^|[^${MATCH_WORD_CHARS}])`
      : ''
    const rightBoundary = !normalizedKeyword.endsWith('*') && isMatchWordChar(normalizedKeyword[normalizedKeyword.length - 1])
      ? `(?=$|[^${MATCH_WORD_CHARS}])`
      : ''
    return new RegExp(`${leftBoundary}${pattern}${rightBoundary}`, 'iu').test(normalizedText)
  } catch {
    return false
  }
}

function filterMatches(row: Record<string, unknown>, filter: GigaChatRulePackFilter) {
  const fieldNames = [filter.field, ...(RULE_FILTER_FIELD_ALIASES[filter.field] ?? [])]
  const fieldName = fieldNames.find((candidate) => Object.prototype.hasOwnProperty.call(row, candidate))
  const left = normalizeMatchText(fieldName ? row[fieldName] : null)
  const right = normalizeMatchText(filter.value)
  return filter.op === 'eq' ? left === right : left !== right
}

export function evaluateRulePacksLocally(values: Record<string, unknown>, rows: Array<Record<string, unknown>>): GigaChatRuleEvaluationResponse {
  const rulePacks = parseRulePacks(values.rule_pack_prompt_notes)
  const evaluations: GigaChatRuleEvaluationRow[] = rows.map((row, rowIndex) => {
    const hits = rulePacks.flatMap((rulePack) => {
      if (!rulePack.enabled) return []
      if (rulePack.filters.length && !rulePack.filters.every((filter) => filterMatches(row, filter))) return []

      const matchedKeywords: string[] = []
      const matchedFields: string[] = []
      rulePack.keywords.forEach((keyword) => {
        let matched = false
        rulePack.source_fields.forEach((fieldName) => {
          if (keywordMatchesText(keyword, row[fieldName])) {
            matched = true
            if (!matchedFields.includes(fieldName)) matchedFields.push(fieldName)
          }
        })
        if (matched) matchedKeywords.push(keyword)
      })

      if (!matchedKeywords.length) return []
      return [{
        code: rulePack.code,
        description: rulePack.description,
        type: rulePack.type,
        matched_keywords: matchedKeywords,
        matched_fields: matchedFields,
        target_tag: rulePack.target_tag,
        target_topic: rulePack.target_topic,
      }]
    })
    return {
      row_index: rowIndex,
      hits,
      suggested_tags: Array.from(new Set(hits.map((hit) => hit.target_tag).filter(Boolean))) as string[],
      suggested_topics: Array.from(new Set(hits.map((hit) => hit.target_topic).filter(Boolean))) as string[],
    }
  })
  return { rule_packs: rulePacks, evaluations }
}

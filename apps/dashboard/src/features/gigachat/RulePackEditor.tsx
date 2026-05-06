import { useMemo, useState } from 'react'
import type { GigaChatRulePack, GigaChatRulePackFilter } from './types'
import { parseRulePacks } from './rulePackMatcher'

function serializeRulePacks(items: GigaChatRulePack[]) {
  return JSON.stringify(items, null, 2)
}

function splitLines(value: string) {
  return value
    .split('\n')
    .map((item) => item.trim())
    .filter(Boolean)
}

function formatRuleAction(rule: GigaChatRulePack) {
  return rule.type === 'assign_tag'
    ? `Tag: ${rule.target_tag || '—'}`
    : `Topic: ${rule.target_topic || '—'}`
}

function emptyFilter(): GigaChatRulePackFilter {
  return { field: '', op: 'eq', value: '' }
}

function emptyRule(): GigaChatRulePack {
  return {
    code: '',
    description: '',
    enabled: true,
    type: 'assign_tag',
    source_fields: [],
    keywords: [],
    filters: [],
    target_tag: null,
    target_topic: null,
  }
}

export function RulePackEditor({
  value,
  onChange,
  availableFields,
  defaultValue,
}: {
  value: unknown
  onChange: (value: string) => void
  availableFields: string[]
  defaultValue?: string
}) {
  const items = useMemo(() => parseRulePacks(value), [value])
  const [editingIndex, setEditingIndex] = useState<number | null>(null)
  const [draft, setDraft] = useState<GigaChatRulePack>(emptyRule())
  const [sourceFieldsText, setSourceFieldsText] = useState('')
  const [keywordsText, setKeywordsText] = useState('')

  const resetDraft = () => {
    setEditingIndex(null)
    setDraft(emptyRule())
    setSourceFieldsText('')
    setKeywordsText('')
  }

  const openEditor = (index: number | null) => {
    if (index === null) {
      resetDraft()
      return
    }
    const rule = items[index]
    setEditingIndex(index)
    setDraft({ ...rule, filters: rule.filters.map((filterItem) => ({ ...filterItem })) })
    setSourceFieldsText(rule.source_fields.join('\n'))
    setKeywordsText(rule.keywords.join('\n'))
  }

  const openCreate = () => {
    setEditingIndex(-1)
    setDraft(emptyRule())
    setSourceFieldsText('')
    setKeywordsText('')
  }

  const saveDraft = () => {
    const nextRule: GigaChatRulePack = {
      ...draft,
      code: draft.code.trim(),
      description: draft.description.trim(),
      source_fields: splitLines(sourceFieldsText),
      keywords: splitLines(keywordsText),
      filters: draft.filters
        .map((filterItem) => ({
          field: filterItem.field.trim(),
          op: filterItem.op,
          value: filterItem.value.trim(),
        }))
        .filter((filterItem) => filterItem.field && filterItem.value),
      target_tag: draft.type === 'assign_tag' ? (draft.target_tag?.trim() || null) : null,
      target_topic: draft.type === 'reclass_topic' ? (draft.target_topic?.trim() || null) : null,
    }
    if (!nextRule.code) return
    const nextItems = [...items]
    if (editingIndex === -1) {
      nextItems.push(nextRule)
    } else if (editingIndex !== null) {
      nextItems[editingIndex] = nextRule
    }
    onChange(serializeRulePacks(nextItems))
    resetDraft()
  }

  const removeItem = (index: number) => {
    onChange(serializeRulePacks(items.filter((_, currentIndex) => currentIndex !== index)))
    if (editingIndex === index) resetDraft()
  }

  const clearAll = () => {
    onChange('[]')
    resetDraft()
  }

  const restoreDefaults = () => {
    if (!defaultValue) return
    onChange(defaultValue)
    resetDraft()
  }

  const isEditing = editingIndex !== null

  return <section className='labeling-panel rule-pack-panel'>
    <div className='labeling-panel-head'>
      <h4>Rule packs</h4>
      <p>Локальные правила, которые до GigaChat ищут совпадения по полям строки, бизнес-фильтрам и keywords, а потом подмешивают их в итоговый prompt как сильную подсказку.</p>
    </div>

    <div className='transport-actions'>
      <button type='button' onClick={openCreate}>+ Добавить правило</button>
      {defaultValue ? <button type='button' onClick={restoreDefaults}>Восстановить DRA/EDU defaults</button> : null}
      <button type='button' onClick={clearAll}>Сбросить все правила</button>
    </div>

    {items.length ? <div className='labeling-rules-list'>
      {items.map((item, index) => <div key={`${item.code}-${index}`} className='labeling-rule-card'>
        <div className='labeling-rule-copy'>
          <div className='labeling-rule-name' title={item.code}>{item.code}</div>
          <div className='labeling-rule-description' title={item.description || 'Без описания'}>{item.description || 'Без описания'}</div>
          <div className='labeling-rule-meta'>
            <span>{item.enabled ? 'Включено' : 'Выключено'}</span>
            <span>{item.type}</span>
            <span>{formatRuleAction(item)}</span>
          </div>
        </div>
        <div className='labeling-rule-actions'>
          <button className='labeling-remove-button' onClick={() => openEditor(index)}>Редактировать</button>
          <button className='labeling-remove-button' onClick={() => removeItem(index)}>Удалить</button>
        </div>
      </div>)}
    </div> : <div className='lab-muted'>Пока rule packs не добавлены.</div>}

    {isEditing ? <div className='sheet-modal-backdrop' onClick={resetDraft}>
      <div className='card sheet-modal labeling-edit-modal rule-pack-edit-modal' onClick={(e) => e.stopPropagation()}>
        <div className='transport-section-head'>
          <div className='transport-section-title'>
            <h3>{editingIndex === -1 ? 'Добавить rule pack' : `Редактировать ${draft.code || 'rule pack'}`}</h3>
            <p>Настройте локальные условия срабатывания правила до отправки строки в GigaChat.</p>
          </div>
          <button className='transport-collapse-button' type='button' onClick={resetDraft}>Закрыть</button>
        </div>

        <div className='labeling-edit-modal-form rule-pack-edit-form'>
          <label className='lab-field'>
            <span>Rule code</span>
            <input type='text' value={draft.code} onChange={(e) => setDraft((current) => ({ ...current, code: e.target.value }))} placeholder='DRA' />
          </label>

          <label className='lab-field'>
            <span>Тип действия</span>
            <select
              value={draft.type}
              onChange={(e) => setDraft((current) => ({
                ...current,
                type: e.target.value === 'reclass_topic' ? 'reclass_topic' : 'assign_tag',
              }))}
            >
              <option value='assign_tag'>assign_tag</option>
              <option value='reclass_topic'>reclass_topic</option>
            </select>
          </label>

          <label className='lab-checkbox'>
            <input
              type='checkbox'
              checked={draft.enabled}
              onChange={(e) => setDraft((current) => ({ ...current, enabled: e.target.checked }))}
            />
            <span><strong>Правило активно</strong></span>
          </label>

          <label className='lab-field wide'>
            <span>Описание</span>
            <textarea value={draft.description} onChange={(e) => setDraft((current) => ({ ...current, description: e.target.value }))} />
          </label>

          <label className='lab-field wide'>
            <span>Source fields</span>
            <textarea
              value={sourceFieldsText}
              onChange={(e) => setSourceFieldsText(e.target.value)}
              placeholder={availableFields.length ? availableFields.join('\n') : 'Во. Описание\nОбр. Результат суммаризации диалога'}
            />
            <small className='lab-field-help'>По одному полю на строку. Только эти поля будут анализироваться локальными правилами.</small>
          </label>

          <label className='lab-field wide'>
            <span>Keywords</span>
            <textarea
              value={keywordsText}
              onChange={(e) => setKeywordsText(e.target.value)}
              placeholder={'транш\nсеместр\nПериод*обучения'}
            />
            <small className='lab-field-help'>По одному ключу на строку. `*` работает как wildcard внутри шаблона.</small>
          </label>

          <div className='rule-pack-filter-list'>
            <div className='rule-pack-filter-head'>
              <strong>Бизнес-фильтры</strong>
              <button type='button' onClick={() => setDraft((current) => ({ ...current, filters: [...current.filters, emptyFilter()] }))}>+ Фильтр</button>
            </div>
            {draft.filters.length ? draft.filters.map((filterItem, index) => <div key={`filter-${index}`} className='rule-pack-filter-row'>
              <input
                type='text'
                value={filterItem.field}
                onChange={(e) => setDraft((current) => ({
                  ...current,
                  filters: current.filters.map((item, currentIndex) => currentIndex === index ? { ...item, field: e.target.value } : item),
                }))}
                placeholder='Трайб'
              />
              <select
                value={filterItem.op}
                onChange={(e) => setDraft((current) => ({
                  ...current,
                  filters: current.filters.map((item, currentIndex) => currentIndex === index ? { ...item, op: e.target.value === 'ne' ? 'ne' : 'eq' } : item),
                }))}
              >
                <option value='eq'>=</option>
                <option value='ne'>!=</option>
              </select>
              <input
                type='text'
                value={filterItem.value}
                onChange={(e) => setDraft((current) => ({
                  ...current,
                  filters: current.filters.map((item, currentIndex) => currentIndex === index ? { ...item, value: e.target.value } : item),
                }))}
                placeholder='ПОТРЕБИТЕЛЬСКИЕ КРЕДИТЫ'
              />
              <button
                type='button'
                onClick={() => setDraft((current) => ({
                  ...current,
                  filters: current.filters.filter((_, currentIndex) => currentIndex !== index),
                }))}
              >
                Удалить
              </button>
            </div>) : <div className='lab-muted'>Фильтров пока нет.</div>}
          </div>

          {draft.type === 'assign_tag' ? <label className='lab-field'>
            <span>Target tag</span>
            <input type='text' value={draft.target_tag ?? ''} onChange={(e) => setDraft((current) => ({ ...current, target_tag: e.target.value }))} placeholder='DRA' />
          </label> : null}

          {draft.type === 'reclass_topic' ? <label className='lab-field wide'>
            <span>Target topic</span>
            <textarea value={draft.target_topic ?? ''} onChange={(e) => setDraft((current) => ({ ...current, target_topic: e.target.value }))} placeholder='Проблема с выдачей очередного транша...' />
          </label> : null}
        </div>

        <div className='lab-settings-actions'>
          <button className='primary' type='button' onClick={saveDraft} disabled={!draft.code.trim()}>
            Сохранить
          </button>
          <button type='button' onClick={resetDraft}>Отмена</button>
        </div>
      </div>
    </div> : null}
  </section>
}

import { useMemo, useState } from 'react'
import { createGigaChatLabSettingsVersion, exportGigaChatRulePacks, importGigaChatRulePacks } from './api'
import type { GigaChatLabSettingsVersionResponse, GigaChatRulePack, GigaChatRulePackFilter } from './types'
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

function splitKeywordLines(value: string) {
  return value
    .split('\n')
    .filter((item) => item.trim())
}

function formatRuleAction(rule: GigaChatRulePack) {
  return rule.type === 'assign_tag'
    ? `Ключевые слова: ${rule.keywords.length ? rule.keywords.join(', ') : '—'}`
    : `Topic: ${rule.target_topic || '—'}`
}

function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  a.remove()
  URL.revokeObjectURL(url)
}

function sourceFieldStatus(rule: GigaChatRulePack, availableFields: string[]) {
  if (!availableFields.length) return { valid: true, missing: [] as string[] }
  if (!rule.source_fields.length) return { valid: false, missing: ['Source fields не выбраны'] }
  const available = new Set(availableFields)
  const ruleFields = [
    ...rule.source_fields,
    ...rule.filters.map((filterItem) => filterItem.field).filter(Boolean),
  ]
  const missing = Array.from(new Set(ruleFields.filter((field) => !available.has(field))))
  return { valid: missing.length === 0, missing }
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
  onPersist,
  persistBusy = false,
  persistError,
  availableFields,
  defaultValue,
  versionId,
  canEdit = false,
  currentVersionTitle,
  currentUserDisplayName,
  onImportComplete,
}: {
  value: unknown
  onChange: (value: string) => void
  onPersist?: (value: string) => void
  persistBusy?: boolean
  persistError?: string | null
  availableFields: string[]
  defaultValue?: string
  versionId?: string
  canEdit?: boolean
  currentVersionTitle?: string
  currentUserDisplayName?: string
  onImportComplete?: (data: GigaChatLabSettingsVersionResponse) => void | Promise<void>
}) {
  const items = useMemo(() => parseRulePacks(value), [value])
  const [editingIndex, setEditingIndex] = useState<number | null>(null)
  const [exchangeOpen, setExchangeOpen] = useState(false)
  const [exchangeMode, setExchangeMode] = useState<'import' | 'export'>('import')
  const [exchangeBusy, setExchangeBusy] = useState(false)
  const [exchangeError, setExchangeError] = useState<string | null>(null)
  const [exchangeMessage, setExchangeMessage] = useState<string | null>(null)
  const [exchangeInputVersion, setExchangeInputVersion] = useState(0)
  const [draft, setDraft] = useState<GigaChatRulePack>(emptyRule())
  const [sourceFieldsText, setSourceFieldsText] = useState('')
  const [keywordsText, setKeywordsText] = useState('')
  const draftSourceFields = useMemo(() => splitLines(sourceFieldsText), [sourceFieldsText])
  const draftHasEmptySourceFields = availableFields.length > 0 && draftSourceFields.length === 0
  const draftMissingSourceFields = useMemo(
    () => availableFields.length ? draftSourceFields.filter((field) => !availableFields.includes(field)) : [],
    [availableFields, draftSourceFields],
  )
  const draftHasMissingSourceFields = draftHasEmptySourceFields || draftMissingSourceFields.length > 0

  const resetDraft = () => {
    setEditingIndex(null)
    setDraft(emptyRule())
    setSourceFieldsText('')
    setKeywordsText('')
  }

  const commitRulePacks = (nextValue: string) => {
    onChange(nextValue)
    onPersist?.(nextValue)
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

  const updateDraftCode = (code: string) => {
    setDraft((current) => {
      const previousCode = current.code.trim()
      const shouldMirrorTarget = current.type === 'assign_tag' && (!current.target_tag || current.target_tag.trim() === previousCode)
      return {
        ...current,
        code,
        target_tag: shouldMirrorTarget ? code : current.target_tag,
      }
    })
  }

  const updateSourceField = (field: string, checked: boolean) => {
    const next = new Set(draftSourceFields)
    if (checked) next.add(field)
    else next.delete(field)
    setSourceFieldsText(Array.from(next).join('\n'))
  }

  const saveDraft = () => {
    if (draftHasMissingSourceFields) return
    const nextRule: GigaChatRulePack = {
      ...draft,
      code: draft.code.trim(),
      description: '',
      source_fields: splitLines(sourceFieldsText),
      keywords: splitKeywordLines(keywordsText),
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
    commitRulePacks(serializeRulePacks(nextItems))
    resetDraft()
  }

  const toggleRuleEnabled = (index: number) => {
    const nextItems = items.map((item, currentIndex) => (
      currentIndex === index && sourceFieldStatus(item, availableFields).valid ? { ...item, enabled: !item.enabled } : item
    ))
    commitRulePacks(serializeRulePacks(nextItems))
  }

  const setAllRulesEnabled = (enabled: boolean) => {
    const nextItems = items.map((item) => {
      const canEnable = sourceFieldStatus(item, availableFields).valid
      return { ...item, enabled: enabled ? canEnable : false }
    })
    commitRulePacks(serializeRulePacks(nextItems))
  }

  const removeItem = (index: number) => {
    commitRulePacks(serializeRulePacks(items.filter((_, currentIndex) => currentIndex !== index)))
    if (editingIndex === index) resetDraft()
  }

  const clearAll = () => {
    commitRulePacks('[]')
    resetDraft()
  }

  const restoreDefaults = () => {
    if (!defaultValue) return
    commitRulePacks(defaultValue)
    resetDraft()
  }

  const openExchange = () => {
    setExchangeOpen(true)
    setExchangeError(null)
    setExchangeMessage(null)
  }

  const closeExchange = () => {
    if (exchangeBusy) return
    setExchangeOpen(false)
    setExchangeError(null)
    setExchangeMessage(null)
  }

  const exportRules = async () => {
    if (!versionId) return
    setExchangeBusy(true)
    setExchangeError(null)
    setExchangeMessage(null)
    try {
      const { blob, filename } = await exportGigaChatRulePacks(versionId)
      downloadBlob(blob, filename || 'rule_packs.xlsx')
      setExchangeMessage('Выгрузка готова. Excel скачан из текущей версии в базе.')
    } catch (error) {
      setExchangeError((error as Error).message || 'Не удалось выгрузить правила.')
    } finally {
      setExchangeBusy(false)
    }
  }

  const importRules = async (file: File | null | undefined) => {
    if (!file || !versionId) return
    setExchangeBusy(true)
    setExchangeError(null)
    setExchangeMessage(null)
    try {
      let targetVersionId = versionId
      let createdCopyTitle = ''
      if (!canEdit) {
        const baseTitle = (currentVersionTitle || versionId || 'Правила').trim()
        const timestamp = new Date().toLocaleString()
        createdCopyTitle = `${baseTitle} · импорт правил`
        const created = await createGigaChatLabSettingsVersion({
          title: createdCopyTitle,
          version_id: null,
          description: `Приватная копия для импорта правил из Excel. Создана ${timestamp}.`,
          status: 'draft',
          visibility: 'private',
          created_by: currentUserDisplayName || '',
          base_version_id: versionId,
        })
        targetVersionId = created.version.version_id
      }
      const data = await importGigaChatRulePacks(targetVersionId, file)
      const imported = parseRulePacks(data.values.rule_pack_prompt_notes).length
      await onImportComplete?.(data)
      setExchangeInputVersion((current) => current + 1)
      setExchangeMessage(
        canEdit
          ? `Правила загружены и сохранены в версии. Импортировано: ${imported}.`
          : `Создана приватная версия "${createdCopyTitle}", правила загружены туда. Импортировано: ${imported}.`,
      )
    } catch (error) {
      setExchangeError((error as Error).message || 'Не удалось загрузить правила.')
    } finally {
      setExchangeBusy(false)
    }
  }

  const isEditing = editingIndex !== null

  return <section className='labeling-panel rule-pack-panel'>
    <div className='labeling-panel-head'>
      <h4>Rule packs</h4>
      <p>Локальные правила, которые до GigaChat ищут совпадения по полям строки, бизнес-фильтрам и keywords, а потом подмешивают их в итоговый prompt как сильную подсказку.</p>
    </div>

    <div className='transport-actions'>
      <button type='button' onClick={openCreate}>+ Добавить правило</button>
      <button type='button' onClick={openExchange}>Загрузить/выгрузить правила</button>
      {defaultValue ? <button type='button' onClick={restoreDefaults}>Восстановить DRA/EDU defaults</button> : null}
      <button type='button' onClick={() => setAllRulesEnabled(true)}>Сделать активными все доступные правила</button>
      <button type='button' onClick={() => setAllRulesEnabled(false)}>Сделать не активными все</button>
      <button type='button' onClick={clearAll}>Сбросить все правила</button>
    </div>
    {persistBusy ? <div className='lab-muted'>Сохраняем правила в файл версии...</div> : null}
    {persistError ? <div className='transport-error'>{persistError}</div> : null}

    {items.length ? <div className='labeling-rules-list'>
      {items.map((item, index) => {
        const status = sourceFieldStatus(item, availableFields)
        const effectiveEnabled = item.enabled && status.valid
        return <div key={`${item.code}-${index}`} className={`labeling-rule-card rule-pack-card ${effectiveEnabled ? 'active' : 'inactive'}${status.valid ? '' : ' missing-fields'}`}>
          <div className='labeling-rule-copy'>
            <div className='labeling-rule-name' title={item.code}>{item.code}</div>
            {status.valid ? null : <div className='rule-pack-missing-fields'>Нет колонок: {status.missing.join(', ')}</div>}
            <div className='labeling-rule-meta'>
              <button
                type='button'
                className={`rule-pack-status-toggle ${effectiveEnabled ? 'active' : 'inactive'}`}
                onClick={() => toggleRuleEnabled(index)}
                disabled={!status.valid}
              >
                {effectiveEnabled ? 'Активно' : 'Не активно'}
              </button>
              <span>{item.type}</span>
              <span>{formatRuleAction(item)}</span>
            </div>
          </div>
          <div className='labeling-rule-actions'>
            <button className='labeling-remove-button' onClick={() => openEditor(index)}>Редактировать</button>
            <button className='labeling-remove-button' onClick={() => removeItem(index)}>Удалить</button>
          </div>
        </div>
      })}
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
            <input type='text' value={draft.code} onChange={(e) => updateDraftCode(e.target.value)} placeholder='DRA' />
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

          <label className='lab-field wide'>
            <span>Source fields</span>
            {availableFields.length ? <div className='rule-source-field-picker'>
              <div className='rule-source-selected'>
                {draftSourceFields.length ? draftSourceFields.map((field) => {
                  const missing = !availableFields.includes(field)
                  return <button
                    key={`selected-${field}`}
                    type='button'
                    className={missing ? 'missing' : ''}
                    onClick={() => updateSourceField(field, false)}
                    title='Нажмите, чтобы убрать поле'
                  >
                    {field}
                  </button>
                }) : <span className='lab-muted'>Поля пока не выбраны.</span>}
              </div>
              <div className='rule-source-options'>
                {availableFields.map((field) => {
                  const selected = draftSourceFields.includes(field)
                  return <button
                    key={field}
                    type='button'
                    className={selected ? 'selected' : ''}
                    onClick={() => updateSourceField(field, true)}
                    disabled={selected}
                  >
                    {field}
                  </button>
                })}
              </div>
            </div> : <textarea
              value={sourceFieldsText}
              onChange={(e) => setSourceFieldsText(e.target.value)}
              placeholder={'Во. Описание\nОбр. Результат суммаризации диалога'}
            />}
            <small className={draftHasMissingSourceFields ? 'transport-error' : 'lab-field-help'}>
              {draftHasEmptySourceFields
                ? 'Выберите хотя бы одну колонку. Без Source fields правило будет выключено.'
                : draftHasMissingSourceFields
                  ? 'В правиле есть поля, которых нет в текущей таблице. Уберите их или выберите существующие колонки.'
                  : availableFields.length
                    ? 'Выберите колонки загруженной таблицы, которые будут анализироваться локальными правилами.'
                    : 'По одному полю на строку. После загрузки Excel здесь появится список колонок.'}
            </small>
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
          <button className='primary' type='button' onClick={saveDraft} disabled={!draft.code.trim() || draftHasMissingSourceFields || persistBusy}>
            {persistBusy ? 'Сохраняем...' : 'Сохранить'}
          </button>
          <button type='button' onClick={resetDraft}>Отмена</button>
        </div>
      </div>
    </div> : null}

    {exchangeOpen ? <div className='sheet-modal-backdrop' onClick={closeExchange}>
      <div className='card sheet-modal rule-pack-exchange-modal' onClick={(e) => e.stopPropagation()}>
        <div className='transport-section-head'>
          <div className='transport-section-title'>
            <h3>Загрузка и выгрузка правил</h3>
            <p>Excel работает с текущей версией настроек. Основные колонки: название тега, ключевые слова и колонка.</p>
          </div>
          <button className='transport-collapse-button' type='button' onClick={closeExchange} disabled={exchangeBusy}>Закрыть</button>
        </div>

        <div className='rule-pack-exchange-tabs' role='tablist' aria-label='Rule packs import export'>
          <button
            type='button'
            className={exchangeMode === 'import' ? 'active' : ''}
            onClick={() => {
              setExchangeMode('import')
              setExchangeError(null)
              setExchangeMessage(null)
            }}
          >
            Загрузка
          </button>
          <button
            type='button'
            className={exchangeMode === 'export' ? 'active' : ''}
            onClick={() => {
              setExchangeMode('export')
              setExchangeError(null)
              setExchangeMessage(null)
            }}
          >
            Выгрузка
          </button>
        </div>

        <div className='rule-pack-exchange-fields'>
          <div>
            <b>Название тега</b>
            <span>Код правила или тег: DRA, IPOTEKA, EDU_RECLASS_TRANCH.</span>
          </div>
          <div>
            <b>Колонка</b>
            <span>Колонка или список колонок, где ищутся ключевые слова.</span>
          </div>
          <div>
            <b>Ключевые слова</b>
            <span>Список ключей через перенос строки или запятую.</span>
          </div>
        </div>

        {exchangeMode === 'import' ? <div className='rule-pack-exchange-panel'>
          <p>{canEdit
            ? 'Загрузка заменит текущий список правил в этой версии и сразу сохранит его в базе.'
            : 'Текущая версия доступна только для просмотра. При загрузке будет создана приватная черновая копия, и правила сохранятся в нее.'}</p>
          <label className={`rule-pack-file-button ${exchangeBusy ? 'disabled' : ''}`}>
            <span>{exchangeBusy ? 'Загружаем...' : canEdit ? 'Выбрать Excel с правилами' : 'Выбрать Excel и создать копию'}</span>
            <input
              key={exchangeInputVersion}
              type='file'
              accept='.xlsx,.xls,.xlsm,.csv'
              disabled={!versionId || exchangeBusy}
              onChange={(event) => importRules(event.target.files?.[0])}
            />
          </label>
        </div> : <div className='rule-pack-exchange-panel'>
          <p>Выгрузка берет текущие правила из базы по выбранной версии и скачивает Excel.</p>
          <button type='button' onClick={exportRules} disabled={!versionId || exchangeBusy}>
            {exchangeBusy ? 'Готовим Excel...' : 'Выгрузить текущие правила из базы'}
          </button>
        </div>}

        {exchangeMessage ? <div className='transport-success'>{exchangeMessage}</div> : null}
        {exchangeError ? <div className='transport-error'>{exchangeError}</div> : null}
      </div>
    </div> : null}
  </section>
}

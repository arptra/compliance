import { useMemo, useState } from 'react'

type LabelingRuleItem = {
  name: string
  description: string
}

function parseRules(raw: unknown): LabelingRuleItem[] {
  const text = String(raw ?? '').trim()
  if (!text) return []
  try {
    const parsed = JSON.parse(text) as unknown
    if (Array.isArray(parsed)) {
      return parsed.map((item) => {
        if (item && typeof item === 'object') {
          const record = item as Record<string, unknown>
          return {
            name: String(record.name ?? '').trim(),
            description: String(record.description ?? '').trim(),
          }
        }
        return { name: '', description: String(item ?? '').trim() }
      }).filter((item) => item.name || item.description)
    }
  } catch {
    return [{ name: 'Импортированный текст', description: text }]
  }
  return []
}

function serializeRules(items: LabelingRuleItem[]) {
  return JSON.stringify(items, null, 2)
}

export function LabelingRulesEditor({
  title,
  description,
  addLabel,
  clearLabel,
  nameLabel,
  value,
  onChange,
  onClear,
}: {
  title: string
  description: string
  addLabel: string
  clearLabel?: string
  nameLabel: string
  value: unknown
  onChange: (value: string) => void
  onClear?: () => void
}) {
  const items = useMemo(() => parseRules(value), [value])
  const [adding, setAdding] = useState(false)
  const [editingIndex, setEditingIndex] = useState<number | null>(null)
  const [draftName, setDraftName] = useState('')
  const [draftDescription, setDraftDescription] = useState('')
  const isEditing = editingIndex !== null

  const resetDraft = () => {
    setDraftName('')
    setDraftDescription('')
    setAdding(false)
    setEditingIndex(null)
  }

  const submitItem = () => {
    const name = draftName.trim()
    const descriptionText = draftDescription.trim()
    if (!name && !descriptionText) return
    if (isEditing) {
      onChange(serializeRules(items.map((item, index) => (
        index === editingIndex ? { name, description: descriptionText } : item
      ))))
      resetDraft()
      return
    }
    onChange(serializeRules([...items, { name, description: descriptionText }]))
    resetDraft()
  }

  const removeItem = (index: number) => {
    onChange(serializeRules(items.filter((_, idx) => idx !== index)))
    if (editingIndex === index) {
      resetDraft()
    }
  }

  const startEdit = (index: number) => {
    const item = items[index]
    setAdding(true)
    setEditingIndex(index)
    setDraftName(item?.name ?? '')
    setDraftDescription(item?.description ?? '')
  }

  return <section className='labeling-panel'>
    <div className='labeling-panel-head'>
      <h4>{title}</h4>
      <p>{description}</p>
    </div>

    <div className='transport-actions'>
      <button onClick={() => {
        if (adding && !isEditing) {
          resetDraft()
          return
        }
        setAdding((current) => {
          const next = !current
          if (!next) {
            resetDraft()
          }
          return next
        })
      }}>{adding && !isEditing ? 'Скрыть форму' : `+ ${addLabel}`}</button>
      {onClear ? <button type='button' onClick={onClear}>{clearLabel ?? 'Сбросить все'}</button> : null}
    </div>

    {adding && !isEditing ? <div className='labeling-entry-form'>
      <input
        type='text'
        value={draftName}
        onChange={(e) => setDraftName(e.target.value)}
        placeholder={nameLabel}
      />
      <input
        type='text'
        value={draftDescription}
        onChange={(e) => setDraftDescription(e.target.value)}
        placeholder='Описание'
      />
      <button onClick={submitItem} disabled={!draftName.trim() && !draftDescription.trim()}>
        Добавить
      </button>
    </div> : null}

    {items.length ? <div className='labeling-rules-list'>
      {items.map((item, index) => <div key={`${item.name}-${index}`} className='labeling-rule-card'>
        <div className='labeling-rule-copy'>
          <div className='labeling-rule-name' title={item.name || nameLabel}>{item.name || nameLabel}</div>
          <div className='labeling-rule-description' title={item.description || 'Без описания'}>{item.description || 'Без описания'}</div>
        </div>
        <div className='labeling-rule-actions'>
          <button className='labeling-remove-button' onClick={() => startEdit(index)}>Редактировать</button>
          <button className='labeling-remove-button' onClick={() => removeItem(index)}>Удалить</button>
        </div>
      </div>)}
    </div> : <div className='lab-muted'>Пока ничего не добавлено.</div>}

    {isEditing ? <div className='sheet-modal-backdrop' onClick={resetDraft}>
      <div className='card sheet-modal labeling-edit-modal' onClick={(e) => e.stopPropagation()}>
        <div className='transport-section-head'>
          <div className='transport-section-title'>
            <h3>Редактировать {nameLabel.toLowerCase()}</h3>
            <p>Изменения сохранятся сразу в правила классификации или тегирования и попадут в итоговый промпт.</p>
          </div>
          <button className='transport-collapse-button' type='button' onClick={resetDraft}>Закрыть</button>
        </div>

        <div className='labeling-edit-modal-form'>
          <label className='lab-field'>
            <span>{nameLabel}</span>
            <input
              type='text'
              value={draftName}
              onChange={(e) => setDraftName(e.target.value)}
              placeholder={nameLabel}
            />
          </label>

          <label className='lab-field wide'>
            <span>Описание</span>
            <textarea
              value={draftDescription}
              onChange={(e) => setDraftDescription(e.target.value)}
              placeholder='Подробное описание и правило выбора'
            />
          </label>
        </div>

        <div className='lab-settings-actions'>
          <button className='primary' type='button' onClick={submitItem} disabled={!draftName.trim() && !draftDescription.trim()}>
            Сохранить
          </button>
          <button type='button' onClick={resetDraft}>Отмена</button>
        </div>
      </div>
    </div> : null}
  </section>
}

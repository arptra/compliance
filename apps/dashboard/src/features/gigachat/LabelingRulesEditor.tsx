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
  const [draftName, setDraftName] = useState('')
  const [draftDescription, setDraftDescription] = useState('')

  const resetDraft = () => {
    setDraftName('')
    setDraftDescription('')
    setAdding(false)
  }

  const addItem = () => {
    const name = draftName.trim()
    const descriptionText = draftDescription.trim()
    if (!name && !descriptionText) return
    onChange(serializeRules([...items, { name, description: descriptionText }]))
    resetDraft()
  }

  const removeItem = (index: number) => {
    onChange(serializeRules(items.filter((_, idx) => idx !== index)))
  }

  return <section className='labeling-panel'>
    <div className='labeling-panel-head'>
      <h4>{title}</h4>
      <p>{description}</p>
    </div>

    <div className='transport-actions'>
      <button onClick={() => setAdding((current) => !current)}>{adding ? 'Скрыть форму' : `+ ${addLabel}`}</button>
      {onClear ? <button type='button' onClick={onClear}>{clearLabel ?? 'Сбросить все'}</button> : null}
    </div>

    {adding ? <div className='labeling-entry-form'>
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
      <button onClick={addItem} disabled={!draftName.trim() && !draftDescription.trim()}>Добавить</button>
    </div> : null}

    {items.length ? <div className='labeling-rules-list'>
      {items.map((item, index) => <div key={`${item.name}-${index}`} className='labeling-rule-card'>
        <div className='labeling-rule-copy'>
          <div className='labeling-rule-name'>{item.name || nameLabel}</div>
          <div className='labeling-rule-description'>{item.description || 'Без описания'}</div>
        </div>
        <button className='labeling-remove-button' onClick={() => removeItem(index)}>Удалить</button>
      </div>)}
    </div> : <div className='lab-muted'>Пока ничего не добавлено.</div>}
  </section>
}

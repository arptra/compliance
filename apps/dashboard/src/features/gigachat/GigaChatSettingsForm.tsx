import type { GigaChatLabSettingField } from './types'

function renderFieldValue(value: unknown) {
  if (typeof value === 'boolean') return value
  if (typeof value === 'number') return value
  if (typeof value === 'string') return value
  if (value == null) return ''
  return String(value)
}

export function GigaChatSettingsForm({
  fields,
  values,
  busy,
  saveError,
  onChange,
  onSave,
}: {
  fields: GigaChatLabSettingField[]
  values: Record<string, unknown>
  busy: boolean
  saveError?: string | null
  onChange: (key: string, value: unknown) => void
  onSave: () => void
}) {
  const grouped = fields.reduce<Record<string, GigaChatLabSettingField[]>>((acc, field) => {
    acc[field.section] = [...(acc[field.section] ?? []), field]
    return acc
  }, {})

  return <div className='lab-settings-grid'>
    {Object.entries(grouped).map(([section, sectionFields]) => <section key={section} className='lab-settings-section'>
      <div className='lab-settings-section-head'>
        <h4>{section}</h4>
      </div>
      <div className='lab-settings-fields'>
        {sectionFields.map((field) => {
          const currentValue = values[field.key]
          if (field.input_type === 'boolean') {
            return <label key={field.key} className='lab-checkbox'>
              <input
                type='checkbox'
                checked={Boolean(currentValue)}
                onChange={(e) => onChange(field.key, e.target.checked)}
              />
              <span>
                <b>{field.label}</b>
                {field.help_text ? <span className='lab-field-help'>{field.help_text}</span> : null}
              </span>
            </label>
          }

          return <label key={field.key} className={`lab-field${field.input_type === 'textarea' ? ' wide' : ''}`}>
            <span>{field.label}</span>
            {field.input_type === 'select' ? <select
              value={String(currentValue ?? '')}
              onChange={(e) => onChange(field.key, e.target.value)}
            >
              {field.options.map((option) => <option key={`${field.key}-${option.value}`} value={option.value}>{option.label}</option>)}
            </select> : null}
            {field.input_type === 'textarea' ? <textarea
              value={String(renderFieldValue(currentValue))}
              onChange={(e) => onChange(field.key, e.target.value)}
            /> : null}
            {field.input_type === 'text' ? <input
              type='text'
              value={String(renderFieldValue(currentValue))}
              onChange={(e) => onChange(field.key, e.target.value)}
            /> : null}
            {field.input_type === 'number' ? <input
              type='number'
              value={String(renderFieldValue(currentValue))}
              onChange={(e) => onChange(field.key, e.target.value === '' ? '' : Number(e.target.value))}
            /> : null}
            {field.help_text ? <span className='lab-field-help'>{field.help_text}</span> : null}
          </label>
        })}
      </div>
    </section>)}

    <div className='lab-settings-actions'>
      <button onClick={onSave} disabled={busy}>{busy ? 'Сохраняем настройки...' : 'Сохранить настройки'}</button>
      {saveError ? <span className='transport-error'>{saveError}</span> : null}
    </div>
  </div>
}

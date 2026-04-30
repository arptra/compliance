import type { GigaChatTransportName, GigaChatTransportStatus } from './types'

function badgeClass(status: GigaChatTransportStatus) {
  if (status.ready) return 'low'
  if (status.configured) return 'medium'
  return 'high'
}

export function GigaChatTransportCard({
  status,
  selected,
  busy,
  collapsed,
  onSelect,
  onProbe,
  onToggle,
}: {
  status: GigaChatTransportStatus
  selected: boolean
  busy: boolean
  collapsed: boolean
  onSelect: (name: GigaChatTransportName) => void
  onProbe: (name: GigaChatTransportName) => void
  onToggle: (name: GigaChatTransportName) => void
}) {
  return <section className={`transport-card${selected ? ' selected' : ''}`}>
    <div className='transport-card-toolbar'>
      <div className='transport-card-head'>
        <div>
          <h3>{status.title}</h3>
          <div className='transport-hint'>{status.description}</div>
        </div>
      </div>
      <div className='transport-card-tools'>
        <span className={`badge ${badgeClass(status)}`}>{status.ready ? 'ready' : status.configured ? 'configured' : 'missing'}</span>
        <button className='transport-collapse-button' onClick={() => onToggle(status.name)}>
          {collapsed ? 'Развернуть' : 'Свернуть'}
        </button>
      </div>
    </div>

    {!collapsed ? <>
      <div className='transport-meta'>
        <div><b>API:</b> <code>{status.base_url}</code></div>
        {status.oauth_url ? <div><b>OAuth:</b> <code>{status.oauth_url}</code></div> : null}
        <div><b>Статус:</b> {status.message}</div>
      </div>

      <div className='transport-artifacts'>
        {status.artifacts.map((artifact) => <div key={`${status.name}-${artifact.label}`} className='transport-artifact'>
          <span>{artifact.label}</span>
          <code>{artifact.path || '—'}</code>
          <span className={`badge ${artifact.exists ? 'low' : 'high'}`}>{artifact.exists ? 'ok' : 'missing'}</span>
        </div>)}
      </div>

      <div className='transport-actions'>
        <button onClick={() => onSelect(status.name)}>{selected ? 'Выбран' : `Выбрать ${status.title}`}</button>
        <button onClick={() => onProbe(status.name)} disabled={busy}>{busy ? 'Проверяем...' : `Проверить ${status.title}`}</button>
      </div>
    </> : null}
  </section>
}

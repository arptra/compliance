type GigaChatProcessingOverlayProps = {
  active: boolean
  minimized: boolean
  title: string
  completed: number
  total: number
  currentLabel?: string
  onBackground: () => void
  onRestore: () => void
}

export function GigaChatProcessingOverlay({
  active,
  minimized,
  title,
  completed,
  total,
  currentLabel,
  onBackground,
  onRestore,
}: GigaChatProcessingOverlayProps) {
  if (!active) return null

  const progressText = `${Math.min(completed, total)} из ${total}`
  const percent = total > 0 ? Math.round((Math.min(completed, total) / total) * 100) : 0

  return <>
    {!minimized ? <div className='sheet-modal-backdrop giga-processing-backdrop'>
      <div className='card giga-processing-modal'>
        <div className='giga-processing-head'>
          <div>
            <h3>{title}</h3>
            <p>Запросы уже отправлены в обработку. Пока окно открыто, экран заблокирован и видно живой прогресс.</p>
          </div>
        </div>

        <div className='giga-processing-body'>
          <div className='spinner giga-processing-spinner' aria-label='gigachat processing' />
          <div className='giga-processing-progress-copy'>
            <div className='giga-processing-progress-title'>{progressText}</div>
            <div className='lab-muted'>{currentLabel || 'Идёт обработка запросов в GigaChat...'}</div>
          </div>
          <div className='giga-processing-progressbar'>
            <div className='giga-processing-progressbar-fill' style={{ width: `${percent}%` }} />
          </div>
        </div>

        <div className='lab-settings-actions'>
          <button type='button' onClick={onBackground}>Делать фоном</button>
        </div>
      </div>
    </div> : null}

    {minimized ? <button className='giga-processing-widget' type='button' onClick={onRestore}>
      <div className='spinner giga-processing-widget-spinner' aria-hidden='true' />
      <div className='giga-processing-widget-copy'>
        <strong>{title}</strong>
        <span>{progressText}</span>
      </div>
    </button> : null}
  </>
}

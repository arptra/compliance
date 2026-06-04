import type { GigaChatWorkbookUploadResponse } from './types'

export function WorkbookSheetsPanel({
  workbook,
  selectedSheetName,
  viewSheetName,
  busySheetName,
  viewBusySheetName,
  onSelect,
  onView,
}: {
  workbook: GigaChatWorkbookUploadResponse | null
  selectedSheetName?: string | null
  viewSheetName?: string | null
  busySheetName?: string | null
  viewBusySheetName?: string | null
  onSelect: (sheetName: string) => void
  onView?: (sheetName: string) => void
}) {
  if (!workbook) return null

  return <section className='workbook-sheets-panel'>
    <div className='workbook-sheets-panel-head'>
      <div>
        <h4>Листы файла</h4>
        <p>Выберите рабочий лист для правил и GigaChat.</p>
      </div>
      <div className='workbook-sheets-counter'>{workbook.sheet_count} листов</div>
    </div>

    <div className='workbook-sheet-list'>
      {workbook.sheets.map((sheet) => {
        const selected = selectedSheetName === sheet.name
        const viewing = viewSheetName === sheet.name
        const busy = busySheetName === sheet.name
        const viewBusy = viewBusySheetName === sheet.name
        return <article key={sheet.name} className={`workbook-sheet-row ${selected ? 'active' : ''} ${viewing ? 'viewing' : ''}`}>
          <div className='workbook-sheet-row-main'>
            <div className='workbook-sheet-title-row'>
              <h4>{sheet.name}</h4>
              {selected ? <span className='workbook-sheet-active-badge'>Рабочий лист</span> : null}
              {viewing ? <span className='workbook-sheet-view-badge'>Открыт</span> : null}
            </div>
            <div className='workbook-sheet-row-meta'>
              <span>Строк: {sheet.rows_total}</span>
              <span>Колонок: {sheet.column_count}</span>
              {sheet.columns.slice(0, 4).map((column) => <span key={`${sheet.name}-${column}`} className='workbook-sheet-column-chip'>{column}</span>)}
            </div>
          </div>
          <div className='workbook-sheet-row-actions'>
            {onView ? <button
              type='button'
              className='workbook-sheet-action-button secondary'
              onClick={() => onView(sheet.name)}
              disabled={viewing || Boolean(viewBusySheetName)}
            >
              {viewBusy ? 'Открываем...' : viewing ? 'Открыт в тетради' : 'Открыть в тетради'}
            </button> : null}
            <button
              type='button'
              className='workbook-sheet-action-button primary'
              onClick={() => onSelect(sheet.name)}
              disabled={selected || Boolean(busySheetName)}
            >
              {busy ? 'Загружаем...' : selected ? 'Выбран для правил' : 'Сделать рабочим'}
            </button>
          </div>
        </article>
      })}
    </div>
  </section>
}

export function WorkbookSheetPickerModal({
  workbook,
  open,
  selectedSheetName,
  busySheetName,
  onClose,
  onSelect,
}: {
  workbook: GigaChatWorkbookUploadResponse | null
  open: boolean
  selectedSheetName?: string | null
  busySheetName?: string | null
  onClose: () => void
  onSelect: (sheetName: string) => void
}) {
  if (!open || !workbook) return null

  return <div className='sheet-modal-backdrop' onClick={onClose}>
    <div className='card sheet-modal' onClick={(e) => e.stopPropagation()}>
      <div className='transport-section-head'>
        <div className='transport-section-title'>
          <h3>Выбор листа</h3>
          <p>В файле <code>{workbook.filename}</code> найдено листов: {workbook.sheet_count}. Выберите, какой лист загрузить в рабочую таблицу.</p>
        </div>
        <button className='transport-collapse-button' onClick={onClose}>Закрыть</button>
      </div>

      <WorkbookSheetsPanel
        workbook={workbook}
        selectedSheetName={selectedSheetName}
        busySheetName={busySheetName}
        onSelect={onSelect}
      />
    </div>
  </div>
}

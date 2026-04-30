import type { GigaChatWorkbookSheetPreview, GigaChatWorkbookUploadResponse } from './types'

function PreviewTable({ sheet }: { sheet: GigaChatWorkbookSheetPreview }) {
  const previewColumns = sheet.columns.slice(0, 6)
  return <div className='sheet-preview-table-wrap'>
    <table className='sheet-preview-table'>
      <thead>
        <tr>{previewColumns.map((column) => <th key={`${sheet.name}-${column}`}>{column}</th>)}</tr>
      </thead>
      <tbody>
        {sheet.preview_rows.map((row, idx) => <tr key={`${sheet.name}-row-${idx}`}>
          {previewColumns.map((column) => <td key={`${sheet.name}-${idx}-${column}`}>{String(row[column] ?? '—')}</td>)}
        </tr>)}
      </tbody>
    </table>
  </div>
}

export function WorkbookSheetPickerModal({
  workbook,
  open,
  busySheetName,
  onClose,
  onSelect,
}: {
  workbook: GigaChatWorkbookUploadResponse | null
  open: boolean
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

      <div className='sheet-preview-grid'>
        {workbook.sheets.map((sheet) => <section key={sheet.name} className='sheet-preview-card'>
          <div className='sheet-preview-head'>
            <div>
              <h4>{sheet.name}</h4>
              <div className='lab-muted'>Строк: {sheet.rows_total} · Колонок: {sheet.column_count}</div>
            </div>
            <button onClick={() => onSelect(sheet.name)} disabled={Boolean(busySheetName)}>
              {busySheetName === sheet.name ? 'Загружаем...' : 'Загрузить лист'}
            </button>
          </div>
          <div className='sheet-preview-columns'>
            {sheet.columns.slice(0, 8).map((column) => <span key={`${sheet.name}-col-${column}`} className='transport-model-pill'>{column}</span>)}
          </div>
          {sheet.preview_rows.length ? <PreviewTable sheet={sheet} /> : <div className='lab-muted'>На листе нет строк для предпросмотра.</div>}
        </section>)}
      </div>
    </div>
  </div>
}

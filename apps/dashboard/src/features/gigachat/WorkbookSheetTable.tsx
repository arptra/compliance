import type { GigaChatWorkbookSheetDataResponse } from './types'
import { useResizableTable, type TableRowClamp } from './useResizableTable'
import { CellHoverPopover, useCellHoverPopover } from './useCellHoverPopover'

type WorkbookRowLimit = 10 | 20 | 100 | 'all'

export function WorkbookSheetTable({
  data,
  onChooseAnotherSheet,
  canChooseAnotherSheet,
  rowLimit,
  onRowLimitChange,
  includedPromptColumns,
  onTogglePromptColumn,
  onRunRow,
  runRowBusyIndex,
  selectedRowKeys,
  allVisibleRowsSelected,
  onToggleRowSelection,
  onToggleAllRows,
  onPickRandomRows,
  onRunSelectedRows,
  batchBusy,
  busy,
}: {
  data: GigaChatWorkbookSheetDataResponse
  onChooseAnotherSheet?: () => void
  canChooseAnotherSheet?: boolean
  rowLimit: WorkbookRowLimit
  onRowLimitChange: (value: WorkbookRowLimit) => void
  includedPromptColumns: string[]
  onTogglePromptColumn: (column: string) => void
  onRunRow: (row: Record<string, unknown>, rowIndex: number) => void
  runRowBusyIndex?: number | null
  selectedRowKeys: string[]
  allVisibleRowsSelected: boolean
  onToggleRowSelection: (rowIndex: number) => void
  onToggleAllRows: (checked: boolean) => void
  onPickRandomRows: () => void
  onRunSelectedRows: () => void
  batchBusy?: boolean
  busy?: boolean
}) {
  const {
    rowClamp,
    setRowClamp,
    startColumnResize,
    resetColumnWidth,
    getColumnStyle,
    cellClampClassName,
    cellClampStyle,
  } = useResizableTable()
  const {
    hoveredCell,
    showCellPopover,
    hideCellPopover,
  } = useCellHoverPopover()

  const renderCellValue = (column: string, value: unknown) => {
    const text = String(value ?? '')
    return <div
      className={cellClampClassName}
      style={cellClampStyle}
      onMouseEnter={(e) => showCellPopover(e, column, text)}
      onMouseLeave={hideCellPopover}
    >
      {text}
    </div>
  }

  return <div className='workbook-table-section'>
    <div className='workbook-table-meta'>
      <div><b>Файл:</b> <code>{data.filename}</code></div>
      <div><b>Лист:</b> <code>{data.sheet_name}</code></div>
      <div><b>Колонки:</b> {data.columns.length}</div>
      <div><b>Показано строк:</b> {data.rendered_rows} из {data.total_rows}</div>
    </div>

    <div className='workbook-table-toolbar'>
      <label className='workbook-row-limit-control'>
        <span>Показывать строк:</span>
        <select
          value={String(rowLimit)}
          disabled={Boolean(busy)}
          onChange={(e) => {
            const value = e.target.value
            if (value === '10' || value === '20' || value === '100') {
              onRowLimitChange(Number(value) as 10 | 20 | 100)
              return
            }
            onRowLimitChange('all')
          }}
        >
          <option value='10'>10</option>
          <option value='20'>20</option>
          <option value='100'>100</option>
          <option value='all'>Все</option>
        </select>
      </label>

      <label className='workbook-row-limit-control'>
        <span>Высота строк:</span>
        <select
          value={String(rowClamp)}
          disabled={Boolean(busy)}
          onChange={(e) => {
            const value = e.target.value
            if (value === '2' || value === '4' || value === '8') {
              setRowClamp(Number(value) as Exclude<TableRowClamp, 'all'>)
              return
            }
            setRowClamp('all')
          }}
        >
          <option value='2'>2 строки</option>
          <option value='4'>4 строки</option>
          <option value='8'>8 строк</option>
          <option value='all'>Полный текст</option>
        </select>
      </label>

      {canChooseAnotherSheet ? <div className='transport-actions'>
        <button onClick={onChooseAnotherSheet}>Выбрать другой лист</button>
      </div> : null}
    </div>

    <div className='transport-actions workbook-batch-actions'>
      <label className='workbook-select-all'>
        <input type='checkbox' checked={allVisibleRowsSelected} disabled={Boolean(busy)} onChange={(e) => onToggleAllRows(e.target.checked)} />
        <span>Выбрать все</span>
      </label>
      <button type='button' onClick={onPickRandomRows} disabled={Boolean(busy)}>Выбрать случайно</button>
      <button type='button' onClick={onRunSelectedRows} disabled={!selectedRowKeys.length || Boolean(batchBusy) || Boolean(busy)}>
        {batchBusy ? 'Отправляем выбранные...' : 'Отправить выбранное в GigaChat для классификации'}
      </button>
      <span className='lab-muted'>Выбрано строк: {selectedRowKeys.length}</span>
    </div>

    <div className='workbook-table-wrap' onMouseLeave={hideCellPopover}>
      <table className='table workbook-table'>
        <thead>
          <tr>
            <th className='workbook-select-head'>
              <input type='checkbox' checked={allVisibleRowsSelected} disabled={Boolean(busy)} onChange={(e) => onToggleAllRows(e.target.checked)} />
            </th>
            <th className='workbook-actions-head'>Действие</th>
            {data.columns.map((column) => {
              const included = includedPromptColumns.includes(column)
              return <th
                key={`head-${column}`}
                className={included ? '' : 'workbook-column-excluded'}
                style={getColumnStyle(column)}
              >
                <div className='workbook-header-cell'>
                  <span>{column}</span>
                  <button
                    className='workbook-column-toggle'
                    onClick={() => onTogglePromptColumn(column)}
                    type='button'
                    disabled={Boolean(busy)}
                  >
                    {included ? 'Убрать из промпта' : 'Добавить в промпт'}
                  </button>
                  <button
                    type='button'
                    className='table-column-resizer'
                    title='Потяните, чтобы изменить ширину колонки. Двойной клик сбрасывает ширину.'
                    aria-label={`Изменить ширину колонки ${column}`}
                    onMouseDown={(e) => startColumnResize(e, column)}
                    onDoubleClick={() => resetColumnWidth(column)}
                  />
                </div>
              </th>
            })}
          </tr>
        </thead>
        <tbody>
          {data.rows.map((row, idx) => <tr
            key={`row-${idx}`}
            className='workbook-table-row'
          >
            <td className='workbook-row-select-cell'>
                <input
                  type='checkbox'
                  checked={selectedRowKeys.includes(`${data.upload_id}:${data.sheet_name}:${idx}`)}
                  disabled={Boolean(busy)}
                  onChange={() => onToggleRowSelection(idx)}
                />
            </td>
            <td className='workbook-row-action-cell'>
              <button
                className='workbook-row-action-button'
                onClick={() => onRunRow(row, idx)}
                disabled={Boolean(busy) || (runRowBusyIndex !== null && runRowBusyIndex !== undefined)}
                type='button'
              >
                {runRowBusyIndex === idx ? 'Отправляем...' : 'Отправить запрос в GigaChat'}
              </button>
            </td>
            {data.columns.map((column) => {
              const included = includedPromptColumns.includes(column)
              return <td
                key={`cell-${idx}-${column}`}
                className={included ? '' : 'workbook-column-excluded'}
                style={getColumnStyle(column)}
              >
                {renderCellValue(column, row[column])}
              </td>
            })}
          </tr>)}
        </tbody>
      </table>
    </div>
    <CellHoverPopover hoveredCell={hoveredCell} />
  </div>
}

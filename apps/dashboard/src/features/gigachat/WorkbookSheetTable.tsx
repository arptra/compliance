import { useMemo, useState } from 'react'
import type { GigaChatRuleEvaluationRow, GigaChatWorkbookSheetDataResponse } from './types'
import { useResizableTable, type TableRowClamp } from './useResizableTable'
import { CellHoverPopover, useCellHoverPopover } from './useCellHoverPopover'
import { useVirtualTableRows } from './useVirtualTableRows'

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
  selectedRowKeySet,
  allVisibleRowsSelected,
  onToggleRowSelection,
  onToggleAllRows,
  onPickRandomRows,
  onRunSelectedRows,
  onRunAllRows,
  onRunSelectedRowsInBackground,
  onExportRows,
  onSelectRuleHitRows,
  onSelectNoRuleHitRows,
  reclassificationRequestsEnabled,
  reclassificationRequestsAvailable,
  onToggleReclassificationRequests,
  batchBusy,
  exportBusy,
  exportPending,
  busy,
  ruleEvaluations,
  rulePackOptions,
  readOnly,
  workingSheetName,
  onMakeCurrentSheetWorking,
  makeCurrentSheetWorkingBusy,
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
  selectedRowKeySet: Set<string>
  allVisibleRowsSelected: boolean
  onToggleRowSelection: (rowIndex: number) => void
  onToggleAllRows: (checked: boolean) => void
  onPickRandomRows: () => void
  onRunSelectedRows: () => void
  onRunAllRows: () => void
  onRunSelectedRowsInBackground?: () => void
  onExportRows: (rows: Array<{ rowIndex: number; row: Record<string, unknown>; evaluation?: GigaChatRuleEvaluationRow }>) => void
  onSelectRuleHitRows: () => void
  onSelectNoRuleHitRows: () => void
  reclassificationRequestsEnabled: boolean
  reclassificationRequestsAvailable: boolean
  onToggleReclassificationRequests: (checked: boolean) => void
  batchBusy?: boolean
  exportBusy?: boolean
  exportPending?: boolean
  busy?: boolean
  ruleEvaluations: Record<number, GigaChatRuleEvaluationRow>
  rulePackOptions: string[]
  readOnly?: boolean
  workingSheetName?: string | null
  onMakeCurrentSheetWorking?: () => void
  makeCurrentSheetWorkingBusy?: boolean
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
  const [showOnlyRuleHits, setShowOnlyRuleHits] = useState(false)
  const [ruleFilter, setRuleFilter] = useState('')
  const tableReadOnly = Boolean(readOnly)

  const ruleOptions = useMemo(
    () => Array.from(new Set([
      ...rulePackOptions,
      ...Object.values(ruleEvaluations).flatMap((evaluation) => evaluation.hits.map((hit) => hit.code)),
    ].filter(Boolean))).sort(),
    [ruleEvaluations, rulePackOptions],
  )
  const visibleRows = useMemo(
    () => data.rows
      .map((row, idx) => ({ row, idx, evaluation: ruleEvaluations[idx] }))
      .filter(({ evaluation }) => {
        if (tableReadOnly) return true
        const hits = evaluation?.hits ?? []
        if (showOnlyRuleHits && !hits.length) return false
        if (ruleFilter && !hits.some((hit) => hit.code === ruleFilter)) return false
        return true
      }),
    [data.rows, ruleEvaluations, ruleFilter, showOnlyRuleHits, tableReadOnly],
  )
  const virtualTable = useVirtualTableRows(visibleRows, rowClamp === 'all' ? 148 : 112)
  const tableColumnCount = (tableReadOnly ? 0 : 7) + data.columns.length
  const selectedRowsForExport = useMemo(
    () => tableReadOnly ? [] : data.rows
      .map((row, idx) => ({ row, idx, evaluation: ruleEvaluations[idx] }))
      .filter(({ idx }) => selectedRowKeySet.has(`${data.upload_id}:${data.sheet_name}:${idx}`)),
    [data.rows, data.sheet_name, data.upload_id, ruleEvaluations, selectedRowKeySet, tableReadOnly],
  )
  const visibleRowsForExport = useMemo(
    () => tableReadOnly ? [] : virtualTable.virtualRows.map(({ item }) => item),
    [tableReadOnly, virtualTable.virtualRows],
  )
  const exportRows = selectedRowsForExport.length ? selectedRowsForExport : visibleRowsForExport

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

    <div className='workbook-table-status-row'>
      <span className={tableReadOnly ? 'workbook-table-readonly-badge' : 'workbook-table-working-badge'}>
        {tableReadOnly ? 'Только просмотр' : 'Рабочий лист'}
      </span>
      {tableReadOnly && workingSheetName ? <span className='lab-muted'>Рабочий лист для правил: <code>{workingSheetName}</code></span> : null}
      {tableReadOnly && onMakeCurrentSheetWorking ? <button
        type='button'
        className='workbook-sheet-action-button primary compact'
        onClick={onMakeCurrentSheetWorking}
        disabled={Boolean(makeCurrentSheetWorkingBusy)}
      >
        {makeCurrentSheetWorkingBusy ? 'Назначаем...' : 'Сделать этот лист рабочим'}
      </button> : null}
    </div>

    <div className='workbook-table-toolbar workbook-table-toolbar-top'>
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
    </div>

    {!tableReadOnly ? <div className='transport-actions workbook-batch-actions'>
      <label className='workbook-reclassification-toggle'>
        <input
          type='checkbox'
          checked={reclassificationRequestsEnabled}
          disabled={Boolean(busy) || !reclassificationRequestsAvailable}
          onChange={(e) => onToggleReclassificationRequests(e.target.checked)}
        />
        <span>Отправлять запросы на переклассификацию</span>
      </label>
      <label className='workbook-select-all'>
        <input type='checkbox' checked={allVisibleRowsSelected} disabled={Boolean(busy)} onChange={(e) => onToggleAllRows(e.target.checked)} />
        <span>Выбрать все</span>
      </label>
      <button type='button' onClick={onPickRandomRows} disabled={Boolean(busy)}>Выбрать случайно</button>
      <button type='button' onClick={onSelectRuleHitRows} disabled={Boolean(busy)}>Выбрать с rule hits</button>
      <button type='button' onClick={onSelectNoRuleHitRows} disabled={Boolean(busy)}>Выбрать без rule hits</button>
      <button
        type='button'
        onClick={() => onExportRows(exportRows.map(({ idx, row, evaluation }) => ({ rowIndex: idx, row, evaluation })))}
        disabled={!exportRows.length || Boolean(exportBusy)}
      >
        {exportPending
          ? 'Выгружаем Excel...'
          : selectedRowsForExport.length
            ? `Выгрузить выбранное в Excel (${selectedRowsForExport.length})`
            : `Выгрузить видимое в Excel (${visibleRowsForExport.length})`}
      </button>
      <button type='button' onClick={onRunSelectedRows} disabled={!selectedRowKeys.length || Boolean(batchBusy) || Boolean(busy)}>
        {batchBusy ? 'Отправляем выбранные...' : 'Отправить выбранное в GigaChat для классификации'}
      </button>
      <button type='button' onClick={onRunAllRows} disabled={!data.rows.length || Boolean(batchBusy) || Boolean(busy)}>
        {batchBusy ? 'Отправляем...' : `Отправить все в GigaChat (${data.rows.length})`}
      </button>
      {onRunSelectedRowsInBackground ? <button type='button' onClick={onRunSelectedRowsInBackground} disabled={!selectedRowKeys.length || Boolean(batchBusy) || Boolean(busy)}>
        Запустить фоном
      </button> : null}
      <span className='lab-muted'>Выбрано строк: {selectedRowKeys.length}</span>
    </div> : null}

    {!tableReadOnly ? <div className='rule-table-controls'>
      <label className='lab-checkbox'>
        <input
          type='checkbox'
          checked={showOnlyRuleHits}
          onChange={(e) => setShowOnlyRuleHits(e.target.checked)}
        />
        <span><strong>Показать только строки с rule hits</strong></span>
      </label>
      <label className='workbook-row-limit-control'>
        <span>Rule pack:</span>
        <select value={ruleFilter} onChange={(e) => setRuleFilter(e.target.value)}>
          <option value=''>Все правила</option>
          {ruleOptions.map((ruleCode) => <option key={ruleCode} value={ruleCode}>{ruleCode}</option>)}
        </select>
      </label>
      <span className='lab-muted'>В таблице: {visibleRows.length} из {data.rows.length}</span>
      <span className='lab-muted'>Отрисовано сейчас: {virtualTable.virtualRows.length}</span>
      <span className='lab-muted'>Пустой rule hit не означает финальное решение. Это только значит, что локальные правила не нашли подсказку; такие строки можно выбрать и отправить в GigaChat.</span>
    </div> : null}

    <div className='workbook-table-wrap' ref={virtualTable.scrollRef} onScroll={virtualTable.onScroll} onMouseLeave={hideCellPopover}>
      <table className='table workbook-table'>
        <thead>
          <tr>
            {!tableReadOnly ? <th className='workbook-select-head'>
              <input type='checkbox' checked={allVisibleRowsSelected} disabled={Boolean(busy)} onChange={(e) => onToggleAllRows(e.target.checked)} />
            </th> : null}
            {!tableReadOnly ? <th className='workbook-actions-head'>Действие</th> : null}
            {!tableReadOnly ? <th style={getColumnStyle('__rule_hits')}>
              <div className='table-simple-header-cell'>
                <span>Rule hits</span>
                <button
                  type='button'
                  className='table-column-resizer'
                  title='Потяните, чтобы изменить ширину колонки. Двойной клик сбрасывает ширину.'
                  aria-label='Изменить ширину колонки Rule hits'
                  onMouseDown={(e) => startColumnResize(e, '__rule_hits')}
                  onDoubleClick={() => resetColumnWidth('__rule_hits')}
                />
              </div>
            </th> : null}
            {!tableReadOnly ? <th style={getColumnStyle('__suggested_actions')}>
              <div className='table-simple-header-cell'>
                <span>Suggested action</span>
                <button
                  type='button'
                  className='table-column-resizer'
                  title='Потяните, чтобы изменить ширину колонки. Двойной клик сбрасывает ширину.'
                  aria-label='Изменить ширину колонки Suggested action'
                  onMouseDown={(e) => startColumnResize(e, '__suggested_actions')}
                  onDoubleClick={() => resetColumnWidth('__suggested_actions')}
                />
              </div>
            </th> : null}
            {!tableReadOnly ? <th style={getColumnStyle('__matched_keywords')}>
              <div className='table-simple-header-cell'>
                <span>Matched keywords</span>
                <button
                  type='button'
                  className='table-column-resizer'
                  title='Потяните, чтобы изменить ширину колонки. Двойной клик сбрасывает ширину.'
                  aria-label='Изменить ширину колонки Matched keywords'
                  onMouseDown={(e) => startColumnResize(e, '__matched_keywords')}
                  onDoubleClick={() => resetColumnWidth('__matched_keywords')}
                />
              </div>
            </th> : null}
            {!tableReadOnly ? <th style={getColumnStyle('__matched_fields')}>
              <div className='table-simple-header-cell'>
                <span>Matched fields</span>
                <button
                  type='button'
                  className='table-column-resizer'
                  title='Потяните, чтобы изменить ширину колонки. Двойной клик сбрасывает ширину.'
                  aria-label='Изменить ширину колонки Matched fields'
                  onMouseDown={(e) => startColumnResize(e, '__matched_fields')}
                  onDoubleClick={() => resetColumnWidth('__matched_fields')}
                />
              </div>
            </th> : null}
            {!tableReadOnly ? <th style={getColumnStyle('__suggested_topic')}>
              <div className='table-simple-header-cell'>
                <span>Suggested topic</span>
                <button
                  type='button'
                  className='table-column-resizer'
                  title='Потяните, чтобы изменить ширину колонки. Двойной клик сбрасывает ширину.'
                  aria-label='Изменить ширину колонки Suggested topic'
                  onMouseDown={(e) => startColumnResize(e, '__suggested_topic')}
                  onDoubleClick={() => resetColumnWidth('__suggested_topic')}
                />
              </div>
            </th> : null}
            {data.columns.map((column) => {
              const included = tableReadOnly || includedPromptColumns.includes(column)
              return <th
                key={`head-${column}`}
                className={tableReadOnly ? '' : `workbook-prompt-column ${included ? 'included' : 'workbook-column-excluded excluded'}`}
                style={getColumnStyle(column)}
                title={tableReadOnly ? undefined : included ? 'Колонка включена в промпт. Клик по заголовку выключит ее.' : 'Колонка выключена из промпта. Клик по заголовку включит ее.'}
                onClick={() => {
                  if (!busy && !tableReadOnly) onTogglePromptColumn(column)
                }}
              >
                <div className='workbook-header-cell'>
                  <span>{column}</span>
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
          {virtualTable.topSpacerHeight ? <tr aria-hidden='true' className='virtual-table-spacer-row'>
            <td colSpan={tableColumnCount} style={{ height: virtualTable.topSpacerHeight }} />
          </tr> : null}
          {virtualTable.virtualRows.map(({ item }) => {
            const { row, idx, evaluation } = item
            const ruleCodes = evaluation?.hits.map((hit) => hit.code).join(', ') || '—'
            const matchedKeywords = Array.from(new Set(evaluation?.hits.flatMap((hit) => hit.matched_keywords) ?? [])).join(', ') || '—'
            const matchedFields = Array.from(new Set(evaluation?.hits.flatMap((hit) => hit.matched_fields) ?? [])).join(', ') || '—'
            const suggestedActions = [
              ...(evaluation?.suggested_tags ?? []).map((item) => `tag:${item}`),
              ...(evaluation?.suggested_topics ?? []).map((item) => `topic:${item}`),
            ].join(', ') || '—'
            const suggestedTopic = evaluation?.suggested_topics?.join(', ') || '—'
            return <tr
              key={`row-${idx}`}
              className='workbook-table-row'
            >
            {!tableReadOnly ? <td className='workbook-row-select-cell'>
                <input
                  type='checkbox'
                  checked={selectedRowKeySet.has(`${data.upload_id}:${data.sheet_name}:${idx}`)}
                  disabled={Boolean(busy)}
                  onChange={() => onToggleRowSelection(idx)}
                />
            </td> : null}
            {!tableReadOnly ? <td className='workbook-row-action-cell'>
              <button
                className='workbook-row-action-button'
                onClick={() => onRunRow(row, idx)}
                disabled={Boolean(busy) || (runRowBusyIndex !== null && runRowBusyIndex !== undefined)}
                type='button'
              >
                {runRowBusyIndex === idx ? 'Отправляем...' : 'Отправить запрос в GigaChat'}
              </button>
            </td> : null}
            {!tableReadOnly ? <td style={getColumnStyle('__rule_hits')}>
              {renderCellValue('Rule hits', ruleCodes)}
            </td> : null}
            {!tableReadOnly ? <td style={getColumnStyle('__suggested_actions')}>
              {renderCellValue('Suggested action', suggestedActions)}
            </td> : null}
            {!tableReadOnly ? <td style={getColumnStyle('__matched_keywords')}>
              {renderCellValue('Matched keywords', matchedKeywords)}
            </td> : null}
            {!tableReadOnly ? <td style={getColumnStyle('__matched_fields')}>
              {renderCellValue('Matched fields', matchedFields)}
            </td> : null}
            {!tableReadOnly ? <td style={getColumnStyle('__suggested_topic')}>
              {renderCellValue('Suggested topic', suggestedTopic)}
            </td> : null}
            {data.columns.map((column) => {
              const included = tableReadOnly || includedPromptColumns.includes(column)
              return <td
                key={`cell-${idx}-${column}`}
                className={included ? '' : 'workbook-column-excluded'}
                style={getColumnStyle(column)}
              >
                {renderCellValue(column, row[column])}
              </td>
            })}
            </tr>
          })}
          {virtualTable.bottomSpacerHeight ? <tr aria-hidden='true' className='virtual-table-spacer-row'>
            <td colSpan={tableColumnCount} style={{ height: virtualTable.bottomSpacerHeight }} />
          </tr> : null}
        </tbody>
      </table>
    </div>
    <div className='workbook-table-toolbar workbook-table-toolbar-bottom'>
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
    <CellHoverPopover hoveredCell={hoveredCell} />
  </div>
}

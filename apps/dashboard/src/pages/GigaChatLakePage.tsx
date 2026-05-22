import { useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useMutation, useQuery } from '@tanstack/react-query'
import {
  GIGACHAT_LAKE_IMPORT_STORAGE_KEY,
  clearRecords,
  deleteRecords,
  searchRecords,
  type GigaChatLakeImportPayload,
  type RecordFilter,
} from '../features/records/api'
import { CellHoverPopover, useCellHoverPopover } from '../features/gigachat/useCellHoverPopover'
import { useResizableTable } from '../features/gigachat/useResizableTable'

type LakeStage = 'raw' | 'rules' | 'gigachat'
type FilterOp = RecordFilter['op']

type FilterDraft = {
  id: string
  column: string
  op: FilterOp
  value: string
  valueTo: string
}

const STAGE_LABELS: Record<LakeStage, string> = {
  raw: 'Исходные',
  rules: 'После правил',
  gigachat: 'После GigaChat',
}

const FILTER_OP_LABELS: Array<{ value: FilterOp; label: string }> = [
  { value: 'contains', label: 'содержит' },
  { value: 'eq', label: 'равно' },
  { value: 'ne', label: 'не равно' },
  { value: 'gte', label: '>=' },
  { value: 'lte', label: '<=' },
  { value: 'between', label: 'между' },
  { value: 'in', label: 'в списке' },
]

const SYSTEM_KEYS = new Set([
  '__record_id',
  '__workspace_id',
  '__file_id',
  '__artifact_id',
  '__uploaded_by_user_id',
  '__uploaded_by_display_name',
  '__uploaded_by_email',
  '__uploaded_by_role',
  '__source_filename',
  '__sheet_name',
  '__source_row_number',
  '__row_hash',
  '__ingested_at',
  '__stage',
  '__period_year',
  '__period_month',
])

function newFilterDraft(): FilterDraft {
  return {
    id: `${Date.now()}-${Math.random().toString(16).slice(2)}`,
    column: '',
    op: 'contains',
    value: '',
    valueTo: '',
  }
}

function toFilter(draft: FilterDraft): RecordFilter | null {
  const column = draft.column.trim()
  if (!column) return null
  if (draft.op === 'between') {
    if (!draft.value.trim() && !draft.valueTo.trim()) return null
    return { column, op: draft.op, value: [draft.value.trim(), draft.valueTo.trim()] }
  }
  if (draft.op === 'in') {
    const values = draft.value.split(',').map((item) => item.trim()).filter(Boolean)
    if (!values.length) return null
    return { column, op: draft.op, value: values }
  }
  if (!draft.value.trim()) return null
  return { column, op: draft.op, value: draft.value.trim() }
}

function rowKey(row: Record<string, unknown>, index: number) {
  return String(row.__record_id ?? `${row.__artifact_id ?? 'row'}:${index}`)
}

function cellText(value: unknown) {
  if (value === null || value === undefined) return ''
  if (typeof value === 'object') return JSON.stringify(value)
  return String(value)
}

function cleanRowForLab(row: Record<string, unknown>, columns: string[]) {
  return Object.fromEntries(
    columns
      .filter((column) => !SYSTEM_KEYS.has(column))
      .map((column) => [column, row[column] ?? '']),
  )
}

export default function GigaChatLakePage() {
  const navigate = useNavigate()
  const lakeTable = useResizableTable(2)
  const {
    hoveredCell,
    showCellPopover,
    hideCellPopover,
  } = useCellHoverPopover()
  const [stage, setStage] = useState<LakeStage>('raw')
  const [limit, setLimit] = useState(100)
  const [offset, setOffset] = useState(0)
  const [filters, setFilters] = useState<FilterDraft[]>([newFilterDraft()])
  const [selectedIds, setSelectedIds] = useState<string[]>([])
  const [mutationMessage, setMutationMessage] = useState('')

  const activeFilters = useMemo(
    () => filters.map(toFilter).filter((item): item is RecordFilter => Boolean(item)),
    [filters],
  )
  const query = useQuery({
    queryKey: ['gigachat-lake-records', stage, activeFilters, limit, offset],
    queryFn: () => searchRecords({ stage, filters: activeFilters, limit, offset }),
    staleTime: 5_000,
  })

  const rows = query.data?.rows ?? []
  const columns = query.data?.columns ?? []
  const selectedIdSet = useMemo(() => new Set(selectedIds), [selectedIds])
  const selectedRows = useMemo(
    () => rows.filter((row, index) => selectedIdSet.has(rowKey(row, index))),
    [rows, selectedIdSet],
  )
  const pageIds = useMemo(() => rows.map(rowKey), [rows])
  const allPageSelected = pageIds.length > 0 && pageIds.every((id) => selectedIdSet.has(id))
  const visibleColumns = columns.slice(0, 36)

  const resetSelection = () => {
    setSelectedIds([])
    setMutationMessage('')
  }

  const changeStage = (nextStage: LakeStage) => {
    setStage(nextStage)
    setOffset(0)
    resetSelection()
  }

  const importSelectedToLab = () => {
    if (!selectedRows.length || !columns.length) return
    const payload: GigaChatLakeImportPayload = {
      upload_id: `lake-import-${Date.now()}`,
      filename: `parquet_${stage}_selection.csv`,
      file_format: 'csv',
      sheet_name: STAGE_LABELS[stage],
      total_rows: selectedRows.length,
      columns,
      rows: selectedRows.map((row) => cleanRowForLab(row, columns)),
      source: {
        stage,
        imported_at: new Date().toISOString(),
      },
    }
    window.sessionStorage.setItem(GIGACHAT_LAKE_IMPORT_STORAGE_KEY, JSON.stringify(payload))
    navigate('/gigachat?import=lake')
  }

  const deleteSelectedRows = useMutation({
    mutationFn: () => {
      const recordIds = selectedRows
        .map((row) => String(row.__record_id ?? '').trim())
        .filter(Boolean)
      return deleteRecords({ stage, record_ids: recordIds })
    },
    onSuccess: async (result) => {
      setSelectedIds([])
      setMutationMessage(result.message || `Удалено строк: ${result.deleted_rows}.`)
      await query.refetch()
    },
  })

  const clearCurrentStage = useMutation({
    mutationFn: () => clearRecords({ stage }),
    onSuccess: async (result) => {
      setSelectedIds([])
      setOffset(0)
      setMutationMessage(result.message || `Слой очищен. Удалено строк: ${result.deleted_rows}.`)
      await query.refetch()
    },
  })

  const isMutatingRecords = deleteSelectedRows.isPending || clearCurrentStage.isPending

  const confirmDeleteSelected = () => {
    if (!selectedRows.length || isMutatingRecords) return
    const ok = window.confirm(`Удалить выбранные строки из слоя "${STAGE_LABELS[stage]}"? Строк: ${selectedRows.length}.`)
    if (ok) deleteSelectedRows.mutate()
  }

  const confirmClearStage = () => {
    if (isMutatingRecords) return
    const ok = window.confirm(`Очистить весь слой "${STAGE_LABELS[stage]}"? Это удалит все parquet-строки текущего слоя.`)
    if (ok) clearCurrentStage.mutate()
  }

  const renderHeader = (key: string, label: string, className = '') => <th
    key={key}
    className={className}
    style={lakeTable.getColumnStyle(key)}
    title={label}
    onMouseEnter={(event) => showCellPopover(event, 'Колонка', label)}
    onMouseLeave={hideCellPopover}
  >
    <div className='lake-header-cell'>
      <span className='lake-header-label'>{label}</span>
      <button
        type='button'
        className='table-column-resizer'
        title={`Изменить ширину колонки ${label}. Двойной клик сбрасывает ширину.`}
        aria-label={`Изменить ширину колонки ${label}`}
        onMouseDown={(event) => lakeTable.startColumnResize(event, key)}
        onDoubleClick={() => lakeTable.resetColumnWidth(key)}
      />
    </div>
  </th>

  const renderCell = (key: string, label: string, value: unknown, className = '') => {
    const text = cellText(value)
    return <td
      className={className}
      style={lakeTable.getColumnStyle(key)}
      onMouseEnter={(event) => showCellPopover(event, label, text)}
      onMouseLeave={hideCellPopover}
    >
      <div className={lakeTable.cellClampClassName} style={lakeTable.cellClampStyle}>{text || '—'}</div>
    </td>
  }

  return <div className='transport-page lake-page'>
    <section className='card transport-hero'>
      <div className='transport-section-head'>
        <div className='transport-section-title'>
          <h2>Parquet-слои GigaChat Lab</h2>
          <p>Превью исходных строк, слоя после правил и слоя после GigaChat. Здесь видны автор загрузки, роль и источник, а выбранные строки можно вернуть в лабораторию разметки.</p>
        </div>
      </div>
      <div className='lake-stage-tabs' role='tablist' aria-label='Parquet layers'>
        {(Object.keys(STAGE_LABELS) as LakeStage[]).map((item) => <button
          key={item}
          type='button'
          className={stage === item ? 'active' : ''}
          onClick={() => changeStage(item)}
        >
          {STAGE_LABELS[item]}
        </button>)}
      </div>
    </section>

    <section className='card transport-result'>
      <div className='transport-section-head'>
        <div className='transport-section-title'>
          <h3>Фильтры</h3>
          <p>Можно фильтровать по любой колонке parquet-файлов. Для списка значений используйте запятую.</p>
        </div>
        <div className='transport-actions'>
          <button type='button' onClick={() => setFilters((current) => [...current, newFilterDraft()])}>Добавить фильтр</button>
          <button
            type='button'
            onClick={() => {
              setFilters([newFilterDraft()])
              setOffset(0)
              resetSelection()
            }}
          >
            Сбросить
          </button>
        </div>
      </div>

      <div className='lake-filter-grid'>
        {filters.map((filter) => <div className='lake-filter-row' key={filter.id}>
          <input
            value={filter.column}
            placeholder='Колонка, например created_at или tag'
            list='lake-column-options'
            onChange={(event) => setFilters((current) => current.map((item) => item.id === filter.id ? { ...item, column: event.target.value } : item))}
          />
          <select
            value={filter.op}
            onChange={(event) => setFilters((current) => current.map((item) => item.id === filter.id ? { ...item, op: event.target.value as FilterOp } : item))}
          >
            {FILTER_OP_LABELS.map((item) => <option key={item.value} value={item.value}>{item.label}</option>)}
          </select>
          <input
            value={filter.value}
            placeholder={filter.op === 'in' ? 'A, B, C' : 'Значение'}
            onChange={(event) => setFilters((current) => current.map((item) => item.id === filter.id ? { ...item, value: event.target.value } : item))}
          />
          {filter.op === 'between' ? <input
            value={filter.valueTo}
            placeholder='До'
            onChange={(event) => setFilters((current) => current.map((item) => item.id === filter.id ? { ...item, valueTo: event.target.value } : item))}
          /> : <span className='lake-filter-spacer' />}
          <button
            type='button'
            onClick={() => setFilters((current) => current.length <= 1 ? [newFilterDraft()] : current.filter((item) => item.id !== filter.id))}
          >
            Удалить
          </button>
        </div>)}
        <datalist id='lake-column-options'>
          {columns.map((column) => <option key={column} value={column} />)}
        </datalist>
      </div>
    </section>

    <section className='card transport-result'>
      <div className='transport-section-head'>
        <div className='transport-section-title'>
          <h3>{STAGE_LABELS[stage]}</h3>
          <p>Найдено строк: {query.data?.total ?? 0}. Движок: {query.data?.engine ?? '...'}</p>
        </div>
        <div className='transport-actions'>
          <button
            type='button'
            onClick={() => query.refetch()}
            disabled={query.isFetching}
          >
            {query.isFetching ? 'Обновляем...' : 'Обновить'}
          </button>
          <button
            type='button'
            className='primary'
            onClick={importSelectedToLab}
            disabled={!selectedRows.length}
          >
            Загрузить выбранное в Lab ({selectedRows.length})
          </button>
        </div>
      </div>

      <div className='lake-summary-grid'>
        <div className='rule-validation-kpi'>
          <span>Слой</span>
          <strong>{STAGE_LABELS[stage]}</strong>
          <small>{stage}</small>
        </div>
        <div className='rule-validation-kpi'>
          <span>Показано</span>
          <strong>{rows.length}</strong>
          <small>из {query.data?.total ?? 0}</small>
        </div>
        <div className='rule-validation-kpi'>
          <span>Выбрано</span>
          <strong>{selectedRows.length}</strong>
          <small>для переноса в лабораторию</small>
        </div>
      </div>

      <div className='lake-danger-zone'>
        <div>
          <strong>Удаление данных</strong>
          <span>Удаляет строки из parquet-файлов текущего слоя. Для исходного слоя также чистится индекс дублей.</span>
        </div>
        <div className='transport-actions'>
          <button
            type='button'
            className='danger'
            onClick={confirmDeleteSelected}
            disabled={!selectedRows.length || isMutatingRecords}
          >
            {deleteSelectedRows.isPending ? 'Удаляем...' : `Удалить выбранные (${selectedRows.length})`}
          </button>
          <button
            type='button'
            className='danger ghost'
            onClick={confirmClearStage}
            disabled={isMutatingRecords || !(query.data?.total ?? 0)}
          >
            {clearCurrentStage.isPending ? 'Очищаем...' : 'Очистить все'}
          </button>
        </div>
        {mutationMessage ? <div className='lake-danger-message'>{mutationMessage}</div> : null}
        {deleteSelectedRows.isError ? <div className='transport-error'>{(deleteSelectedRows.error as Error).message}</div> : null}
        {clearCurrentStage.isError ? <div className='transport-error'>{(clearCurrentStage.error as Error).message}</div> : null}
      </div>

      <div className='transport-actions lake-table-actions'>
        <label className='workbook-select-all'>
          <input
            type='checkbox'
            checked={allPageSelected}
            disabled={!pageIds.length}
            onChange={(event) => {
              setSelectedIds((current) => {
                const next = new Set(current)
                if (event.target.checked) pageIds.forEach((id) => next.add(id))
                else pageIds.forEach((id) => next.delete(id))
                return Array.from(next)
              })
            }}
          />
          <span>Выбрать страницу</span>
        </label>
        <label className='workbook-row-limit-control'>
          <span>Строк на странице:</span>
          <select
            value={limit}
            onChange={(event) => {
              setLimit(Number(event.target.value))
              setOffset(0)
              resetSelection()
            }}
          >
            <option value={50}>50</option>
            <option value={100}>100</option>
            <option value={250}>250</option>
            <option value={500}>500</option>
          </select>
        </label>
        <button type='button' onClick={() => setOffset(Math.max(0, offset - limit))} disabled={offset <= 0}>Назад</button>
        <button type='button' onClick={() => setOffset(offset + limit)} disabled={offset + limit >= (query.data?.total ?? 0)}>Дальше</button>
        <span className='lab-muted'>Страница с {offset + 1} по {offset + rows.length}</span>
      </div>

      {query.isError ? <div className='transport-error'>{(query.error as Error).message}</div> : null}
      {!query.isLoading && !rows.length ? <div className='lab-muted'>В этом слое пока нет parquet-строк или фильтры ничего не нашли.</div> : null}
      {query.isLoading ? <div className='lab-muted'>Загружаем parquet-превью...</div> : null}

      {rows.length ? <div className='sheet-preview-table-wrap lake-table-wrap' onMouseLeave={hideCellPopover}>
        <table className='table sheet-preview-table lake-table'>
          <thead>
            <tr>
              {renderHeader('__select', 'Выбор', 'workbook-select-head lake-select-head')}
              {renderHeader('__author', 'Автор')}
              {renderHeader('__role', 'Роль')}
              {renderHeader('__filename', 'Файл')}
              {renderHeader('__sheet', 'Лист')}
              {renderHeader('__source_row', 'Строка')}
              {visibleColumns.map((column) => renderHeader(column, column))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, index) => {
              const id = rowKey(row, index)
              return <tr key={id}>
                <td className='workbook-row-select-cell lake-select-cell' style={lakeTable.getColumnStyle('__select')}>
                  <input
                    type='checkbox'
                    checked={selectedIdSet.has(id)}
                    onChange={() => {
                      setSelectedIds((current) => {
                        const next = new Set(current)
                        if (next.has(id)) next.delete(id)
                        else next.add(id)
                        return Array.from(next)
                      })
                    }}
                  />
                </td>
                {renderCell('__author', 'Автор', row.__uploaded_by_display_name || row.__uploaded_by_email || row.__uploaded_by_user_id)}
                {renderCell('__role', 'Роль', row.__uploaded_by_role)}
                {renderCell('__filename', 'Файл', row.__source_filename)}
                {renderCell('__sheet', 'Лист', row.__sheet_name)}
                {renderCell('__source_row', 'Строка', row.__source_row_number)}
                {visibleColumns.map((column) => renderCell(column, column, row[column]))}
              </tr>
            })}
          </tbody>
        </table>
      </div> : null}
      <CellHoverPopover hoveredCell={hoveredCell} />
    </section>
  </div>
}

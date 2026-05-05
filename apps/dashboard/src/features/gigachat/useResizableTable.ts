import { useEffect, useRef, useState, type CSSProperties, type MouseEvent as ReactMouseEvent } from 'react'

export type TableRowClamp = 2 | 4 | 8 | 'all'

type ResizeState = {
  columnKey: string
  startX: number
  startWidth: number
}

export function useResizableTable(defaultRowClamp: TableRowClamp = 4) {
  const [rowClamp, setRowClamp] = useState<TableRowClamp>(defaultRowClamp)
  const [columnWidths, setColumnWidths] = useState<Record<string, number>>({})
  const resizeStateRef = useRef<ResizeState | null>(null)

  useEffect(() => {
    const handleMouseMove = (event: MouseEvent) => {
      const state = resizeStateRef.current
      if (!state) return
      const nextWidth = Math.max(96, state.startWidth + (event.clientX - state.startX))
      setColumnWidths((current) => ({ ...current, [state.columnKey]: nextWidth }))
    }

    const finishResize = () => {
      if (!resizeStateRef.current) return
      resizeStateRef.current = null
      document.body.style.cursor = ''
      document.body.style.userSelect = ''
    }

    window.addEventListener('mousemove', handleMouseMove)
    window.addEventListener('mouseup', finishResize)
    return () => {
      window.removeEventListener('mousemove', handleMouseMove)
      window.removeEventListener('mouseup', finishResize)
      finishResize()
    }
  }, [])

  const startColumnResize = (event: ReactMouseEvent<HTMLElement>, columnKey: string) => {
    event.preventDefault()
    event.stopPropagation()
    const header = event.currentTarget.closest('th') as HTMLTableCellElement | null
    const startWidth = Math.max(header?.getBoundingClientRect().width ?? 180, 96)
    resizeStateRef.current = {
      columnKey,
      startX: event.clientX,
      startWidth,
    }
    document.body.style.cursor = 'col-resize'
    document.body.style.userSelect = 'none'
  }

  const resetColumnWidth = (columnKey: string) => {
    setColumnWidths((current) => {
      if (!(columnKey in current)) return current
      const next = { ...current }
      delete next[columnKey]
      return next
    })
  }

  const getColumnStyle = (columnKey: string): CSSProperties | undefined => {
    const width = columnWidths[columnKey]
    if (!width) return undefined
    return {
      width,
      minWidth: width,
      maxWidth: width,
    }
  }

  const cellClampClassName = rowClamp === 'all'
    ? 'workbook-cell-compact workbook-cell-expanded'
    : 'workbook-cell-compact'

  const cellClampStyle = rowClamp === 'all'
    ? undefined
    : ({ '--cell-lines': String(rowClamp) } as CSSProperties)

  return {
    rowClamp,
    setRowClamp,
    startColumnResize,
    resetColumnWidth,
    getColumnStyle,
    cellClampClassName,
    cellClampStyle,
  }
}

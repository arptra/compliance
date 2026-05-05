import { useEffect, useState, type CSSProperties, type MouseEvent as ReactMouseEvent } from 'react'

type HoveredCellState = {
  label: string
  text: string
  style: CSSProperties
}

function buildPopoverStyle(element: HTMLElement): CSSProperties {
  const rect = element.getBoundingClientRect()
  const width = Math.min(520, Math.max(280, window.innerWidth - 32))
  const left = Math.min(Math.max(16, rect.left), window.innerWidth - width - 16)
  const showAbove = rect.bottom > window.innerHeight - 260
  const top = showAbove
    ? Math.max(16, rect.top - 228)
    : Math.min(window.innerHeight - 244, rect.bottom + 8)

  return {
    position: 'fixed',
    left,
    top,
    width,
  }
}

export function useCellHoverPopover() {
  const [hoveredCell, setHoveredCell] = useState<HoveredCellState | null>(null)

  useEffect(() => {
    if (!hoveredCell) return
    const hide = () => setHoveredCell(null)
    window.addEventListener('scroll', hide, true)
    window.addEventListener('resize', hide)
    return () => {
      window.removeEventListener('scroll', hide, true)
      window.removeEventListener('resize', hide)
    }
  }, [hoveredCell])

  const showCellPopover = (event: ReactMouseEvent<HTMLElement>, label: string, value: unknown) => {
    const text = String(value ?? '').trim() || '—'
    const element = event.currentTarget as HTMLElement
    setHoveredCell({
      label,
      text,
      style: buildPopoverStyle(element),
    })
  }

  const hideCellPopover = () => setHoveredCell(null)

  return {
    hoveredCell,
    showCellPopover,
    hideCellPopover,
  }
}

export function CellHoverPopover({ hoveredCell }: { hoveredCell: HoveredCellState | null }) {
  if (!hoveredCell) return null

  return <div className='cell-hover-popover' style={hoveredCell.style}>
    <div className='cell-hover-popover-label'>{hoveredCell.label}</div>
    <div className='cell-hover-popover-text'>{hoveredCell.text}</div>
  </div>
}

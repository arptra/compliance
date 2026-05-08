import { useCallback, useEffect, useMemo, useRef, useState, type UIEvent } from 'react'

export type VirtualTableRow<T> = {
  item: T
  index: number
}

export function useVirtualTableRows<T>(
  items: T[],
  estimateRowHeight = 96,
  overscan = 8,
) {
  const scrollRef = useRef<HTMLDivElement | null>(null)
  const [scrollTop, setScrollTop] = useState(0)
  const [viewportHeight, setViewportHeight] = useState(640)

  const onScroll = useCallback((event: UIEvent<HTMLDivElement>) => {
    setScrollTop(event.currentTarget.scrollTop)
  }, [])

  useEffect(() => {
    const element = scrollRef.current
    if (!element) return

    const updateHeight = () => setViewportHeight(element.clientHeight || 640)
    updateHeight()

    const observer = new ResizeObserver(updateHeight)
    observer.observe(element)
    return () => observer.disconnect()
  }, [])

  useEffect(() => {
    const element = scrollRef.current
    if (!element) return
    const maxScrollTop = Math.max(items.length * estimateRowHeight - viewportHeight, 0)
    if (element.scrollTop > maxScrollTop) {
      element.scrollTop = maxScrollTop
      setScrollTop(maxScrollTop)
    }
  }, [estimateRowHeight, items.length, viewportHeight])

  const range = useMemo(() => {
    const startIndex = Math.max(Math.floor(scrollTop / estimateRowHeight) - overscan, 0)
    const visibleCount = Math.ceil(viewportHeight / estimateRowHeight) + overscan * 2
    const endIndex = Math.min(startIndex + visibleCount, items.length)

    return {
      startIndex,
      endIndex,
      topSpacerHeight: startIndex * estimateRowHeight,
      bottomSpacerHeight: Math.max((items.length - endIndex) * estimateRowHeight, 0),
    }
  }, [estimateRowHeight, items.length, overscan, scrollTop, viewportHeight])

  const virtualRows = useMemo(
    () => items
      .slice(range.startIndex, range.endIndex)
      .map((item, index) => ({ item, index: range.startIndex + index })),
    [items, range.endIndex, range.startIndex],
  )

  return {
    scrollRef,
    onScroll,
    virtualRows,
    topSpacerHeight: range.topSpacerHeight,
    bottomSpacerHeight: range.bottomSpacerHeight,
  }
}

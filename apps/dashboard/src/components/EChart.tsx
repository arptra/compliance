import { useEffect, useRef } from 'react'
import * as echarts from 'echarts'

export type EChartProps = {
  option: echarts.EChartsCoreOption
  height?: number | string
}

export function EChart({ option, height = 320 }: EChartProps) {
  const ref = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    if (!ref.current) return
    const chart = echarts.init(ref.current)
    chart.setOption(option)

    const onResize = () => chart.resize()
    window.addEventListener('resize', onResize)

    return () => {
      window.removeEventListener('resize', onResize)
      chart.dispose()
    }
  }, [option])

  return <div ref={ref} style={{ width: '100%', height }} />
}

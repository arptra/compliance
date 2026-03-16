import { Link, Outlet, useLocation } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { FilterBar } from './FilterBar'
import { useRunAction } from '../hooks/useRunAction'
import { apiGet } from '../lib/api'

const items = [
  ['/overview', 'Overview'], ['/categories', 'Categories'], ['/timeseries', 'Timeseries'],
  ['/pattern-fit', 'Pattern Fit'], ['/pattern-monitor', 'Pattern Monitor'], ['/reports', 'Reports'], ['/settings', 'Settings']
]

function StartupLoader() {
  return <div className='startup-loader'>
    <div className='spinner' aria-label='loading' />
    <div className='loader-title'>Загрузка данных из parquet…</div>
  </div>
}

export function Layout() {
  const loc = useLocation()
  const runViz = useRunAction('/api/runs/viz-build')
  const boot = useQuery({ queryKey: ['boot-datasets'], queryFn: () => apiGet('/api/meta/datasets') })

  if (boot.isLoading) return <StartupLoader />

  return <div className="layout">
    <aside className="sidebar">{items.map(([to,label]) => <Link key={to} to={to} style={{fontWeight: loc.pathname===to?700:400}}>{label}</Link>)}</aside>
    <main className="main">
      <header className="header"><div>Interactive Dashboard</div><button onClick={() => runViz.mutate({tag:'latest'})}>Run viz-build</button></header>
      <FilterBar />
      <section className="content"><Outlet /></section>
    </main>
  </div>
}

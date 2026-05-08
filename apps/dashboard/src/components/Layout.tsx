import { Link, Outlet, useLocation } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { FilterBar } from './FilterBar'
import { apiGet } from '../lib/api'

const items = [
  ['/overview', 'Overview'], ['/categories', 'Categories'], ['/timeseries', 'Timeseries'],
  ['/preparation', 'Preparation'], ['/gigachat', 'GigaChat Lab'], ['/gigachat/background', '  Фоновые задачи'], ['/pattern-fit', 'Pattern Fit'], ['/pattern-monitor', 'Pattern Monitor'], ['/review-dataset', 'Review Dataset'], ['/model-quality', 'Model Quality'], ['/parquet-viewer', 'Parquet Viewer'], ['/reports', 'Reports'], ['/settings', 'Settings']
]

function StartupLoader() {
  return <div className='startup-loader'>
    <div className='spinner' aria-label='loading' />
    <div className='loader-title'>Загрузка данных из parquet…</div>
  </div>
}

export function Layout() {
  const loc = useLocation()
  const boot = useQuery({ queryKey: ['boot-datasets'], queryFn: () => apiGet('/api/meta/datasets') })
  const showFilters = !loc.pathname.startsWith('/gigachat')

  if (boot.isLoading) return <StartupLoader />

  return <div className="layout">
    <aside className="sidebar">{items.map(([to,label]) => <Link key={to} to={to} className={to.includes('/background') ? 'sidebar-subitem' : ''} style={{fontWeight: loc.pathname===to?700:400}}>{label}</Link>)}</aside>
    <main className="main">
      <header className="header"><div>Interactive Dashboard</div></header>
      {showFilters ? <FilterBar /> : null}
      <section className="content"><Outlet /></section>
    </main>
  </div>
}

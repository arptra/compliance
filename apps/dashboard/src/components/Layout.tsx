import { Link, Outlet, useLocation } from 'react-router-dom'
import { FilterBar } from './FilterBar'

const items = [
  ['/overview', 'Overview'], ['/categories', 'Categories'], ['/timeseries', 'Timeseries'],
  ['/preparation', 'Preparation'], ['/gigachat', 'GigaChat Lab'], ['/gigachat/background', '  Фоновые задачи'], ['/pattern-fit', 'Pattern Fit'], ['/pattern-monitor', 'Pattern Monitor'], ['/review-dataset', 'Review Dataset'], ['/model-quality', 'Model Quality'], ['/parquet-viewer', 'Parquet Viewer'], ['/reports', 'Reports'], ['/settings', 'Settings']
]

export function Layout() {
  const loc = useLocation()
  const showFilters = !loc.pathname.startsWith('/gigachat')

  return <div className="layout">
    <aside className="sidebar">{items.map(([to,label]) => <Link key={to} to={to} className={to.includes('/background') ? 'sidebar-subitem' : ''} style={{fontWeight: loc.pathname===to?700:400}}>{label}</Link>)}</aside>
    <main className="main">
      <header className="header"><div>Interactive Dashboard</div></header>
      {showFilters ? <FilterBar /> : null}
      <section className="content"><Outlet /></section>
    </main>
  </div>
}

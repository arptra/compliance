import { Link, Outlet, useLocation } from 'react-router-dom'
import { FilterBar } from './FilterBar'
import { useRunAction } from '../hooks/useRunAction'

const items = [
  ['/overview', 'Overview'], ['/categories', 'Categories'], ['/timeseries', 'Timeseries'],
  ['/pattern-fit', 'Pattern Fit'], ['/pattern-monitor', 'Pattern Monitor'], ['/reports', 'Reports'], ['/settings', 'Settings']
]

export function Layout() {
  const loc = useLocation()
  const runViz = useRunAction('/api/runs/viz-build')
  return <div className="layout">
    <aside className="sidebar">{items.map(([to,label]) => <Link key={to} to={to} style={{fontWeight: loc.pathname===to?700:400}}>{label}</Link>)}</aside>
    <main className="main">
      <header className="header"><div>Interactive Dashboard</div><button onClick={() => runViz.mutate({tag:'latest'})}>Run viz-build</button></header>
      <FilterBar />
      <section className="content"><Outlet /></section>
    </main>
  </div>
}

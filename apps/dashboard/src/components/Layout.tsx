import { useEffect, useState } from 'react'
import { Link, Outlet, useLocation } from 'react-router-dom'
import { FilterBar } from './FilterBar'
import { useAuth } from '../features/auth/AuthContext'

const items = [
  { to: '/overview', label: 'Overview', short: 'O' },
  { to: '/categories', label: 'Categories', short: 'C' },
  { to: '/timeseries', label: 'Timeseries', short: 'T' },
  { to: '/preparation', label: 'Preparation', short: 'P' },
  { to: '/gigachat', label: 'GigaChat Lab', short: 'G' },
  { to: '/gigachat/lake', label: 'Загруженные данные', short: 'ЗД', sub: true },
  { to: '/gigachat/background', label: 'Фоновые задачи', short: 'Ф', sub: true },
  { to: '/pattern-fit', label: 'Pattern Fit', short: 'PF' },
  { to: '/pattern-monitor', label: 'Pattern Monitor', short: 'PM' },
  { to: '/review-dataset', label: 'Review Dataset', short: 'R' },
  { to: '/model-quality', label: 'Model Quality', short: 'M' },
  { to: '/parquet-viewer', label: 'Parquet Viewer', short: 'PV' },
  { to: '/reports', label: 'Reports', short: 'R' },
  { to: '/settings', label: 'Settings', short: 'S' },
]

const SIDEBAR_COLLAPSED_KEY = 'complaints-dashboard-sidebar-collapsed'

function MenuIcon({ collapsed }: { collapsed: boolean }) {
  return <svg viewBox='0 0 24 24' aria-hidden='true' className='sidebar-button-icon'>
    {collapsed
      ? <path d='M9 6l6 6-6 6' fill='none' stroke='currentColor' strokeWidth='2' strokeLinecap='round' strokeLinejoin='round' />
      : <path d='M15 6l-6 6 6 6' fill='none' stroke='currentColor' strokeWidth='2' strokeLinecap='round' strokeLinejoin='round' />}
  </svg>
}

function UserIcon() {
  return <svg viewBox='0 0 24 24' aria-hidden='true' className='sidebar-profile-icon'>
    <circle cx='12' cy='8' r='4' fill='none' stroke='currentColor' strokeWidth='2' />
    <path d='M4 21a8 8 0 0 1 16 0' fill='none' stroke='currentColor' strokeWidth='2' strokeLinecap='round' />
  </svg>
}

export function Layout() {
  const loc = useLocation()
  const { user, logout } = useAuth()
  const [sidebarCollapsed, setSidebarCollapsed] = useState(() => window.localStorage.getItem(SIDEBAR_COLLAPSED_KEY) === '1')
  const showFilters = !loc.pathname.startsWith('/gigachat') && loc.pathname !== '/profile'
  const userName = user?.display_name || user?.email || 'Профиль'

  useEffect(() => {
    window.localStorage.setItem(SIDEBAR_COLLAPSED_KEY, sidebarCollapsed ? '1' : '0')
  }, [sidebarCollapsed])

  return <div className={`layout ${sidebarCollapsed ? 'sidebar-collapsed' : ''}`}>
    <aside className="sidebar" aria-label='Main navigation'>
      <div className='sidebar-top'>
        <button
          type='button'
          className='sidebar-collapse-button'
          onClick={() => setSidebarCollapsed((current) => !current)}
          aria-label={sidebarCollapsed ? 'Развернуть меню' : 'Свернуть меню'}
          title={sidebarCollapsed ? 'Развернуть меню' : 'Свернуть меню'}
        >
          <MenuIcon collapsed={sidebarCollapsed} />
          <span>Меню</span>
        </button>
      </div>
      <Link
        to='/profile'
        title='Профиль'
        className={`sidebar-profile-link ${loc.pathname === '/profile' ? 'active' : ''}`}
      >
        <UserIcon />
        <span>
          <strong>{userName}</strong>
          <small>{user?.role || 'user'}</small>
        </span>
      </Link>
      <nav className='sidebar-nav'>
        {items.map((item) => {
          const active = loc.pathname === item.to
          return <Link
            key={item.to}
            to={item.to}
            title={item.label}
            className={`sidebar-link ${item.sub ? 'sidebar-subitem' : ''} ${active ? 'active' : ''}`}
          >
            <span className='sidebar-link-short'>{item.short}</span>
            <span className='sidebar-link-label'>{item.label}</span>
          </Link>
        })}
      </nav>
    </aside>
    <main className="main">
      <header className="header">
        <div>Interactive Dashboard</div>
        <div className='user-menu'>
          <Link to='/profile'>{userName}</Link>
          <span className='lab-muted'>{user?.role}</span>
          <button type='button' onClick={() => void logout()}>Выйти</button>
        </div>
      </header>
      {showFilters ? <FilterBar /> : null}
      <section className="content"><Outlet /></section>
    </main>
  </div>
}

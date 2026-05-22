import { Navigate, Outlet, useLocation } from 'react-router-dom'
import { useAuth } from './AuthContext'

export function AuthGate() {
  const { user, loading } = useAuth()
  const location = useLocation()

  if (loading) return <div className='auth-shell'><div className='card auth-card'>Проверяем сессию...</div></div>
  if (!user) return <Navigate to='/login' replace state={{ from: location.pathname }} />
  return <Outlet />
}

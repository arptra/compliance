import { Navigate, createBrowserRouter } from 'react-router-dom'
import { Layout } from '../components/Layout'
import GigaChatPage from '../pages/GigaChatPage'
import GigaChatBackgroundTasksPage from '../pages/GigaChatBackgroundTasksPage'
import GigaChatLakePage from '../pages/GigaChatLakePage'
import ProfilePage from '../pages/ProfilePage'
import AuthPage from '../pages/AuthPage'
import { AuthGate } from '../features/auth/AuthGate'

export const router = createBrowserRouter([
  { path: '/login', element: <AuthPage mode='login' /> },
  { path: '/register', element: <AuthPage mode='register' /> },
  { element: <AuthGate />, children: [
    { path: '/', element: <Layout />, children: [
      { index: true, element: <Navigate to='/gigachat' replace /> },
      { path: 'gigachat', element: <GigaChatPage /> },
      { path: 'gigachat/lake', element: <GigaChatLakePage /> },
      { path: 'gigachat/background', element: <GigaChatBackgroundTasksPage /> },
      { path: 'profile', element: <ProfilePage /> },
      { path: '*', element: <Navigate to='/gigachat' replace /> }
    ] }
  ] }
])

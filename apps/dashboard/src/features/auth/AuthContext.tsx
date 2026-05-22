import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from 'react'
import { getAuthToken, setAuthToken } from '../../lib/api'
import { getCurrentUser, login as loginRequest, logout as logoutRequest, register as registerRequest, type AuthUser } from './api'

type AuthContextValue = {
  user: AuthUser | null
  loading: boolean
  login: (email: string, password: string) => Promise<void>
  register: (email: string, password: string, displayName: string) => Promise<void>
  logout: () => Promise<void>
  setCurrentUser: (user: AuthUser) => void
}

const AuthContext = createContext<AuthContextValue | null>(null)

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<AuthUser | null>(null)
  const [loading, setLoading] = useState(Boolean(getAuthToken()))

  useEffect(() => {
    if (!getAuthToken()) return
    let alive = true
    setLoading(true)
    getCurrentUser()
      .then((data) => {
        if (alive) setUser(data)
      })
      .catch(() => {
        setAuthToken(null)
        if (alive) setUser(null)
      })
      .finally(() => {
        if (alive) setLoading(false)
      })
    return () => {
      alive = false
    }
  }, [])

  const value = useMemo<AuthContextValue>(() => ({
    user,
    loading,
    login: async (email, password) => {
      const data = await loginRequest(email, password)
      setAuthToken(data.access_token)
      setUser(data.user)
    },
    register: async (email, password, displayName) => {
      const data = await registerRequest(email, password, displayName)
      setAuthToken(data.access_token)
      setUser(data.user)
    },
    logout: async () => {
      try {
        await logoutRequest()
      } finally {
        setAuthToken(null)
        setUser(null)
      }
    },
    setCurrentUser: (nextUser) => setUser(nextUser),
  }), [loading, user])

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

export function useAuth() {
  const context = useContext(AuthContext)
  if (!context) throw new Error('useAuth must be used inside AuthProvider')
  return context
}

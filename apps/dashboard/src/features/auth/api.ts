import { apiGet, apiPatch, apiPost } from '../../lib/api'

export type AuthUser = {
  id: string
  email: string
  display_name: string
  first_name: string
  last_name: string
  role: string
  workspace_id: string
  workspace_role: string
}

export type AuthTokenResponse = {
  access_token: string
  token_type: string
  expires_at: string
  user: AuthUser
}

export function getCurrentUser() {
  return apiGet<AuthUser>('/api/auth/me')
}

export function login(email: string, password: string) {
  return apiPost<AuthTokenResponse>('/api/auth/login', { email, password })
}

export function register(email: string, password: string, display_name: string) {
  return apiPost<AuthTokenResponse>('/api/auth/register', { email, password, display_name })
}

export function updateProfile(payload: { first_name: string; last_name: string; display_name?: string }) {
  return apiPatch<AuthUser>('/api/auth/me', payload)
}

export function changePassword(current_password: string, new_password: string) {
  return apiPost<{ ok: boolean }>('/api/auth/me/password', { current_password, new_password })
}

export function logout() {
  return apiPost<{ ok: boolean }>('/api/auth/logout', {})
}

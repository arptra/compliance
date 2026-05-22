import { FormEvent, useState } from 'react'
import { Link, Navigate, useLocation, useNavigate } from 'react-router-dom'
import { useAuth } from '../features/auth/AuthContext'

export default function AuthPage({ mode }: { mode: 'login' | 'register' }) {
  const { user, login, register } = useAuth()
  const navigate = useNavigate()
  const location = useLocation()
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [displayName, setDisplayName] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const from = (location.state as { from?: string } | null)?.from || '/overview'

  if (user) return <Navigate to={from} replace />

  const submit = async (event: FormEvent) => {
    event.preventDefault()
    setBusy(true)
    setError(null)
    try {
      if (mode === 'login') await login(email, password)
      else await register(email, password, displayName)
      navigate(from, { replace: true })
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Не удалось выполнить вход.')
    } finally {
      setBusy(false)
    }
  }

  return <div className='auth-shell'>
    <form className='card auth-card' onSubmit={submit}>
      <div className='auth-head'>
        <h1>{mode === 'login' ? 'Вход' : 'Регистрация'}</h1>
        <p>Доступ к файлам, настройкам и parquet-артефактам привязывается к пользователю.</p>
      </div>
      <label className='lab-field'>
        <span>Email</span>
        <input type='email' value={email} onChange={(event) => setEmail(event.target.value)} required autoComplete='email' />
      </label>
      {mode === 'register' ? <label className='lab-field'>
        <span>Имя</span>
        <input value={displayName} onChange={(event) => setDisplayName(event.target.value)} autoComplete='name' />
      </label> : null}
      <label className='lab-field'>
        <span>Пароль</span>
        <input type='password' value={password} onChange={(event) => setPassword(event.target.value)} required minLength={6} autoComplete={mode === 'login' ? 'current-password' : 'new-password'} />
      </label>
      {error ? <div className='transport-error'>{error}</div> : null}
      <div className='auth-actions'>
        <button className='primary' type='submit' disabled={busy}>
          {busy ? 'Подождите...' : mode === 'login' ? 'Войти' : 'Создать аккаунт'}
        </button>
        {mode === 'login'
          ? <Link to='/register'>Создать аккаунт</Link>
          : <Link to='/login'>Уже есть аккаунт</Link>}
      </div>
    </form>
  </div>
}

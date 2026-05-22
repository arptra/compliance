import { FormEvent, useEffect, useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { changePassword, updateProfile } from '../features/auth/api'
import { useAuth } from '../features/auth/AuthContext'

function formatProfileError(error: unknown, fallback: string) {
  if (!(error instanceof Error)) return fallback
  try {
    const parsed = JSON.parse(error.message) as { detail?: string }
    return parsed.detail || fallback
  } catch {
    return error.message || fallback
  }
}

export default function ProfilePage() {
  const { user, setCurrentUser } = useAuth()
  const [firstName, setFirstName] = useState(user?.first_name || '')
  const [lastName, setLastName] = useState(user?.last_name || '')
  const [currentPassword, setCurrentPassword] = useState('')
  const [newPassword, setNewPassword] = useState('')
  const [repeatPassword, setRepeatPassword] = useState('')
  const [profileMessage, setProfileMessage] = useState('')
  const [passwordMessage, setPasswordMessage] = useState('')

  useEffect(() => {
    setFirstName(user?.first_name || '')
    setLastName(user?.last_name || '')
  }, [user?.first_name, user?.last_name])

  const saveProfile = useMutation({
    mutationFn: () => updateProfile({ first_name: firstName, last_name: lastName }),
    onSuccess: (updatedUser) => {
      setCurrentUser(updatedUser)
      setProfileMessage('Профиль сохранен.')
    },
  })

  const savePassword = useMutation({
    mutationFn: () => changePassword(currentPassword, newPassword),
    onSuccess: () => {
      setCurrentPassword('')
      setNewPassword('')
      setRepeatPassword('')
      setPasswordMessage('Пароль изменен.')
    },
  })

  const submitProfile = (event: FormEvent) => {
    event.preventDefault()
    setProfileMessage('')
    saveProfile.mutate()
  }

  const submitPassword = (event: FormEvent) => {
    event.preventDefault()
    setPasswordMessage('')
    if (newPassword !== repeatPassword) {
      setPasswordMessage('Новый пароль и повтор не совпадают.')
      return
    }
    savePassword.mutate()
  }

  return <div className='profile-page'>
    <section className='card transport-hero profile-hero'>
      <div>
        <h2>Профиль</h2>
        <p>Личные данные, роль и смена пароля для текущего пользователя.</p>
      </div>
      <div className='profile-role-card'>
        <span>Роль</span>
        <strong>{user?.role || 'user'}</strong>
        <small>Workspace: {user?.workspace_id || 'default'} · {user?.workspace_role || user?.role || 'user'}</small>
      </div>
    </section>

    <section className='profile-grid'>
      <form className='card profile-card' onSubmit={submitProfile}>
        <div className='transport-section-title'>
          <h3>Личные данные</h3>
          <p>Имя и фамилия будут показываться в интерфейсе и рядом с загруженными файлами.</p>
        </div>
        <label className='lab-field'>
          <span>Email</span>
          <input value={user?.email || ''} disabled />
        </label>
        <label className='lab-field'>
          <span>Имя</span>
          <input value={firstName} onChange={(event) => setFirstName(event.target.value)} autoComplete='given-name' />
        </label>
        <label className='lab-field'>
          <span>Фамилия</span>
          <input value={lastName} onChange={(event) => setLastName(event.target.value)} autoComplete='family-name' />
        </label>
        <div className='transport-actions'>
          <button type='submit' className='primary' disabled={saveProfile.isPending}>
            {saveProfile.isPending ? 'Сохраняем...' : 'Сохранить профиль'}
          </button>
        </div>
        {profileMessage ? <div className='profile-success'>{profileMessage}</div> : null}
        {saveProfile.isError ? <div className='transport-error'>{formatProfileError(saveProfile.error, 'Не удалось сохранить профиль.')}</div> : null}
      </form>

      <form className='card profile-card' onSubmit={submitPassword}>
        <div className='transport-section-title'>
          <h3>Пароль</h3>
          <p>Для безопасности нужен текущий пароль. Новый пароль должен быть не короче 6 символов.</p>
        </div>
        <label className='lab-field'>
          <span>Текущий пароль</span>
          <input type='password' value={currentPassword} onChange={(event) => setCurrentPassword(event.target.value)} required autoComplete='current-password' />
        </label>
        <label className='lab-field'>
          <span>Новый пароль</span>
          <input type='password' value={newPassword} onChange={(event) => setNewPassword(event.target.value)} required minLength={6} autoComplete='new-password' />
        </label>
        <label className='lab-field'>
          <span>Повторите новый пароль</span>
          <input type='password' value={repeatPassword} onChange={(event) => setRepeatPassword(event.target.value)} required minLength={6} autoComplete='new-password' />
        </label>
        <div className='transport-actions'>
          <button type='submit' className='primary' disabled={savePassword.isPending}>
            {savePassword.isPending ? 'Меняем...' : 'Изменить пароль'}
          </button>
        </div>
        {passwordMessage ? <div className={passwordMessage.includes('не совпадают') ? 'transport-error' : 'profile-success'}>{passwordMessage}</div> : null}
        {savePassword.isError ? <div className='transport-error'>{formatProfileError(savePassword.error, 'Не удалось изменить пароль.')}</div> : null}
      </form>
    </section>
  </div>
}
